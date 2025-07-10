import cupy as cp
from app.tasks.TA52_Modeler.TA52_0_ModelerOrchestrator import ModelerJob


import os
os.environ["CUML_LOG_LEVEL"] = "DEBUG"

import cuml  # must come after the env var is set



def hdb_validation_classifier(job: ModelerJob, Z: cp.ndarray, y: cp.ndarray, label: str, frac: float) -> ModelerJob:
    """
    Run HDBSCAN clustering for a given label and fraction.
    Evaluates both default and adaptive cluster sizes.

    Parameters
    ----------
    job : ModelerJob
        The active pipeline job.
    Z : cp.ndarray
        Dimensionality-reduced embedding matrix.
    y : cp.ndarray
        Encoded labels for the index column.
    label : str
        The name of the index column being evaluated.
    frac : float
        Dimensionality reduction retention level.

    Returns
    -------
    ModelerJob
        The same job object with validation results and stats updated.
    """
    from app.tasks.TA52_Modeler.TA52_C_Validator import PARAMS_DICT

    bootstrap = getattr(job.input, "bootstrap_iteration", 0)
    scope = job.input.scope
    min_cluster_default = PARAMS_DICT.get("HDBSCAN_MIN_CLUSTER_DEFAULT", 20)

    # Variant 1: Default cluster size
    run_hdbscan_variant(
        job=job,
        Z=Z,
        y=y,
        label=label,
        frac=frac,
        bootstrap=bootstrap,
        min_cluster_size=min_cluster_default,
        min_samples = min_cluster_default,
        variant_name="default"
    )

    def adaptive_min_samples_gpu(
        min_cluster_size: int,
        n_samples: int,
        max_samples: int = 50,
        max_sample_ratio: float = 0.3
    ) -> int:
        """
        Adaptively computes min_samples based on min_cluster_size, capped to a fraction of n_samples.

        Parameters
        ----------
        min_cluster_size : int
            The minimum cluster size from the dataset.
        n_samples : int
            Total number of samples (e.g., Z_gpu.shape[0]).
        max_samples : int, optional
            The asymptotic max min_samples value (default: 50).
        max_sample_ratio : float, optional
            Maximum allowed min_samples as a fraction of n_samples (default: 0.3 = 30%).

        Returns
        -------
        int
            Safe, scaled min_samples (≥2, ≤max_sample_ratio * n_samples).
        """
        import cupy as cp

        scale = 0.01  # conservative slope
        raw_value = max_samples * (1 - cp.exp(-scale * min_cluster_size))
        cap = int(n_samples * max_sample_ratio)
        value = cp.minimum(raw_value, cap)

        return max(2, int(cp.asnumpy(value)))




    percentiles = PARAMS_DICT.get("HDBSCAN_PERCENTILES", [5, 10, 20, 30, 40, 50, 75])
    unique_vals, counts = cp.unique(y, return_counts=True)
    valid_counts = counts[counts > 0]  # Optional, but keeps logic robust

    for p in percentiles:
        # Compute GPU-based percentile
        perc_val = cp.percentile(valid_counts, p)
        min_cluster_size = max(int(float(perc_val)), 2)
        min_samples = adaptive_min_samples_gpu(min_cluster_size, PARAMS_DICT["HDBSCAN_MIN_SAMPLES_CEILING"])



        variant_name = f"adaptive_p{p:02d}"
        if min_cluster_size > PARAMS_DICT["HDBSCAN_MIN_CLUSTER_DEFAULT"]:
            run_hdbscan_variant(
                job,
                Z,
                y,
                label,
                frac,
                bootstrap=bootstrap,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                variant_name=variant_name
            )
        else:
            job.input.fail_trail.mark_validation(
                bootstrap=bootstrap,
                label=label,
                frac=frac,
                model=f"hdbscan_{variant_name}",
                status="skipped",
                error=f"min_cluster_size={min_cluster_size} < threshold"
            )

    return job

def run_hdbscan_variant(
    job: ModelerJob,
    Z: cp.ndarray,
    y: cp.ndarray,
    label: str,
    frac: float,
    bootstrap: int,
    min_cluster_size: int,
    min_samples: int,
    variant_name: str,
) -> ModelerJob:
    """
    Run a single variant of HDBSCAN clustering and record metrics.

    Parameters
    ----------
    job : ModelerJob
        The pipeline job object being updated.
    Z : cp.ndarray
        Embedding matrix after dimensionality reduction.
    y : cp.ndarray
        Ground-truth encoded label vector.
    label : str
        Name of the hierarchy level (e.g. genus, species).
    frac : float
        Fraction of retained dimensions.
    bootstrap : int
        Bootstrap iteration ID.
    min_cluster_size : int
        Minimum cluster size for HDBSCAN.
    variant_name : str
        Label for the variant (e.g., "default", "adaptive").

    Returns
    -------
    ModelerJob
        The job with results, stats, and fail_trail updates.
    """

    from cuml.cluster import HDBSCAN

    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        silhouette_score
    )
    from app.tasks.TA52_Modeler.TA52_C_Validator import _GPU_LOCK
    from app.tasks.TA52_Modeler.TA52_C_Validator import PARAMS_DICT

    import numpy as np
    import scipy.stats
    import cupy as cp
    import time
    import logging
    import gc

    t0 = time.time()
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    try:
        with _GPU_LOCK:
            Z_gpu: cp.ndarray = cp.ascontiguousarray(Z.copy())
            if Z_gpu.dtype != cp.float32:
                Z_gpu = Z_gpu.astype(cp.float32)

            # --- Input Sanity Checks ---
            has_nan = cp.isnan(Z_gpu).any()
            has_inf = not cp.isfinite(Z_gpu).all()
            constant_cols = int(cp.any(cp.std(Z_gpu, axis=0) == 0))
            zero_dim = Z_gpu.shape[1] == 0
            too_small = Z_gpu.shape[0] < PARAMS_DICT["HDBSCAN_MIN_SAMPLE_THRESHOLD"]
            too_small_cluster = min_cluster_size >= Z_gpu.shape[0]

            if any([has_nan, has_inf, zero_dim, too_small, too_small_cluster, constant_cols]):
                reason = " → ".join([
                    f"NaN={has_nan}",
                    f"Inf={has_inf}",
                    f"ZeroDim={zero_dim}",
                    f"TooSmall={too_small}",
                    f"MinCluster>{Z_gpu.shape[0]}",
                    f"ConstCols={constant_cols}"
                ])
                logging.debug1(f"[HDBSCAN-{variant_name}] Skipped '{label}' frac={frac:.2f}6 | {reason}")
                job.input.fail_trail.mark_validation(
                    bootstrap=bootstrap,
                    label=label,
                    frac=frac,
                    model=f"hdbscan_{variant_name}",
                    status="skipped",
                    error=reason
                )
                return job
            
            unique, counts = cp.unique(y, return_counts=True)
            smallest_class = counts.min()
            largest_class = counts.max()
            entropy = float(scipy.stats.entropy(counts.get()))

            logging.debug2(f"[HDBSCAN-{variant_name}] Running '{label}' frac={frac:.2f} | "
                        f"min_cluster_size={min_cluster_size} min_samples={min_samples}"
                        f" | Z_gpu.shape={Z_gpu.shape} | n_unique={unique.size} | "
                        f"smallest class={smallest_class} largest class={largest_class} "
                        f"entropy={entropy:.4f}")
            if unique.size < 3 or entropy <= 0.9:
                logging.warning(f"[HDBSCAN-{variant_name}] Skipped: "
                                f"Insufficient diversity (n_unique={unique.size}, entropy={entropy:.4f})")
                job.input.fail_trail.mark_validation(
                    bootstrap=bootstrap,
                    label=label,
                    frac=frac,
                    model=f"hdbscan_{variant_name}",
                    status="skipped",
                    error=f"Too few classes or low entropy (n_unique={unique.size}, entropy={entropy:.4f})"
                )
                return job

            cp.cuda.Device(0).synchronize()
            clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples = min_samples, prediction_data=False)
            cluster_labels: cp.ndarray = clusterer.fit_predict(Z_gpu)
            cp.cuda.runtime.deviceSynchronize()

        # --- Post-process Metrics ---
        cluster_labels_host = cp.asnumpy(cluster_labels)
        y_host = cp.asnumpy(y)
        valid_mask = cluster_labels_host >= 0

        if valid_mask.sum() < 2:
            logging.debug1(f"[HDBSCAN-{variant_name}] <2 valid clusters for '{label}' frac={frac:.2f}")
            job.input.fail_trail.mark_validation(
                bootstrap=bootstrap,
                label=label,
                frac=frac,
                model=f"hdbscan_{variant_name}",
                status="skipped",
                error="<2 valid clusters"
            )
            return job

        ari = adjusted_rand_score(y_host[valid_mask], cluster_labels_host[valid_mask])
        nmi = normalized_mutual_info_score(y_host[valid_mask], cluster_labels_host[valid_mask])
        #sil = silhouette_score(cp.asnumpy(Z_gpu), cluster_labels_host)

        # --- Store Results ---
        job.attrs.validation_results_dict \
            .setdefault(bootstrap, {}) \
            .setdefault(label, {}) \
            .setdefault(frac, {}) \
            .update({
                f"ari_{variant_name}": ari,
                f"nmi_{variant_name}": nmi,
                #f"silhouette_{variant_name}": sil,   # To enhance PErformance
            })

        job.stats \
            .setdefault("validation", {}) \
            .setdefault(bootstrap, {}) \
            .setdefault(label, {}) \
            .setdefault(frac, {}) \
            [f"hdbscan_time_{variant_name}"] = time.time() - t0

        logging.debug1(
            f"[HDBSCAN-{variant_name}] '{label}' frac={frac:.2f} | "
            #f"ARI={ari:.4f} NMI={nmi:.4f} Sil={sil:.4f}"
            f"ARI={ari:.4f} NMI={nmi:.4f}"
        )

        job.input.fail_trail.mark_validation(
            bootstrap=bootstrap,
            label=label,
            frac=frac,
            model=f"hdbscan_{variant_name}",
            status="passed"
        )


        

    except Exception as exc:
        logging.warning(f"[HDBSCAN-{variant_name}] failed '{label}' frac={frac:.2f}: {exc}")
        job.input.fail_trail.mark_validation(
            bootstrap=bootstrap,
            label=label,
            frac=frac,
            model=f"hdbscan_{variant_name}",
            status="failed",
            error=str(exc)
        )


    finally:
        for var in ['Z_gpu', 'cluster_labels', 'cluster_labels_host', 'y_host', 'valid_mask']:
            if var in locals():
                del locals()[var]
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        time.sleep(0.1)  # Allow GPU to settle

        return job
