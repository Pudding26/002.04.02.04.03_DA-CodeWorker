from __future__ import annotations

"""TA52_C_Validator
====================
Validator stage for the TA‑52 modelling pipeline.  
Runs supervised (Random‑Forest, Logistic Regression) and unsupervised (HDBSCAN) evaluation
on PCA‑transformed embeddings produced by `TA52_B_Modeler`.  
The class also persists a **results table** and **UMAP cluster map** for downstream analytics.

Design goals
------------
* **Single‑GPU safety** – All GPU kernels are executed inside a global lock to guarantee
  that only one CUDA context is active at a time.  This makes the whole module safe even
  if the orchestrator is extended to spawn multiple worker threads in the future.
* **CPU‑only hand‑off** – No `cupy.ndarray` or other GPU object ever leaves the validator.
  Pandas DataFrames (or plain Python types) are produced before storing or returning.
* **Robust logging** – Every important step is logged with the project‑wide `DEBUG‑x`
  convention.
* **Public‑ready** – Complete doc‑strings, type hints and a clean dependency list so the
  module can be published as part of the TA‑52 open‑source release.
"""
import os
os.environ["CUML_LOG_LEVEL"] = "error"


from collections import defaultdict
from threading import Lock
import logging
import time
from typing import Dict, List, Any, Tuple, Optional

import cupy as cp
import numpy as np
import pandas as pd
from cuml.ensemble import RandomForestClassifier
from cuml.linear_model import LogisticRegression
from cuml.cluster import HDBSCAN
from cuml.decomposition import PCA
from cuml.manifold import UMAP
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

# ---------------------------------------------------------------------------
# Project‑specific storage helpers (replace the import path with your package)
# ---------------------------------------------------------------------------
from app.utils.common.app.utils.dataModels.Jobs.ModelerJob import ModelerJob



# ---------------------------------------------------------------------------

__all__ = ["TA52_C_Validator"]

# Global GPU lock – guarantees single‑threaded CUDA execution
_GPU_LOCK: Lock = Lock()

# Hierarchical label columns used for scope/level selection in embeddings
INDEX_COLS: List[str] = [
    "family",
    "genus",
    "species",
    "sourceID",
    "specimenID",
    "sampleID",
    "stackID",
    "shotID",
]


class TA52_C_Validator:
    """Run model validation, persist metrics & embeddings."""

    # ---------------------------------------------------------------------
    # Public entry‑point
    # ---------------------------------------------------------------------
    @staticmethod
    def run(job: ModelerJob) -> ModelerJob:
        """Execute classifiers + clustering on every PCA sweep inside *job*.

        Parameters
        ----------
        job : ModelerJob
            The job produced by *TA52_B_Modeler* containing
            `job.attrs.multi_pca_results` (dict of frac→embedding matrices)
            and encoded label columns in `job.attrs.data_num`.
        """
        logging.debug5("[VALIDATOR] Starting validation process …")

        data_sweep: Dict[float, Dict[str, Any]] = job.attrs.multi_pca_results
        index_cols: List[str] = job.attrs.blacklist["index_cols"]
        if job.attrs.results_cupy is None:
            job.attrs.results_cupy = {}


        timings = {"RandomForest": 0.0, "KNN": 0.0, "HDBSCAN": 0.0}

        for frac, result in data_sweep.items():
            Z: cp.ndarray = result["Z_total"]
            for label in index_cols:
                idx = job.attrs.encoder.cols[label]
                y: cp.ndarray = job.attrs.data_num[:, idx]
                
                if not isinstance(job.attrs.uniques, dict):
                    job.attrs.uniques = {}

                if label not in job.attrs.uniques:
                    job.attrs.uniques[label] = {}

                job.attrs.uniques[label][frac] = TA52_C_Validator._compute_uniques_for_scope(job, y, label)

                
                # Skip labels with <2 classes
                if len(cp.unique(y)) < 2:
                    logging.debug1(
                        f"[VALIDATOR] Skip label '{label}' (<2 classes) @ frac={frac:.2f}"
                    )
                    continue

                # ------- Supervised ----------
                clf_t0 = time.time()
                TA52_C_Validator._classify_and_store(job, Z, y, label, frac)
                clf_t1 = time.time()

                # ------- Clustering ----------
                TA52_C_Validator._knn_and_store(job, Z, y, label, frac)

                TA52_C_Validator._hdbscan_and_store_wrapper(job, Z, y, label, frac)

        # Summary log + persistence
        logging.debug2("Adding results to job attributes...")
        job.attrs.result_df = TA52_C_Validator.store_job_result(job)
        job.attrs.featureClusterMap = TA52_C_Validator.store_ClusterEmbeddings(job)
        TA52_C_Validator._log_summary(job)    

        logging.debug5("[VALIDATOR] Validation process completed.")
        return job

    # ------------------------------------------------------------------
    # Internal helpers – supervised evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _knn_and_store(job: ModelerJob, Z: cp.ndarray, y: cp.ndarray, label: str, frac: float, n_neighbors: int = 5) -> None:
        """Run 5‑fold KNN classification using cuML, store accuracies."""
        from cuml.neighbors import KNeighborsClassifier
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score

        t0 = time.time()

        Z_host, y_host = cp.asnumpy(Z), cp.asnumpy(y)
        if len(Z_host) < 20:
            logging.debug1(
                f"[KNN] Skip label='{label}' @ frac={frac:.2f} – too few rows ({len(Z_host)})"
            )
            return

        skf = StratifiedKFold(n_splits=5, shuffle=True)
        knn_scores = []

        for train_idx, test_idx in skf.split(Z_host, y_host):
            Z_train, Z_test = Z_host[train_idx], Z_host[test_idx]
            y_train, y_test = y_host[train_idx], y_host[test_idx]

            with _GPU_LOCK:
                knn = KNeighborsClassifier(n_neighbors=n_neighbors)
                knn.fit(cp.asarray(Z_train), cp.asarray(y_train))
                y_pred_knn = knn.predict(cp.asarray(Z_test))
                cp.cuda.runtime.deviceSynchronize()

            acc = float(accuracy_score(y_test, cp.asnumpy(y_pred_knn)))
            knn_scores.append(acc)

        mean_knn = round(np.mean(knn_scores), 4)
        logging.debug1(
            f"[KNN] label='{label}' frac={frac:.2f} KNN={mean_knn:.4f}"
        )
        job.attrs.results_cupy.setdefault(label, {}).setdefault(frac, {})
        job.attrs.results_cupy[label][frac]['knn_acc'] = mean_knn

        knn_time = time.time() - t0
        job.stats.setdefault("validation", {}).setdefault(label, {}).setdefault(frac, {})["knn_time"] = knn_time



    @staticmethod
    def _classify_and_store(job: ModelerJob, Z: cp.ndarray, y: cp.ndarray, label: str, frac: float) -> None:
        """Run lightweight 5‑fold Random Forest classification on GPU and store accuracies & timings."""
        from cuml.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The number of bins.*", category=UserWarning)

            if Z.shape[0] < 20:
                logging.debug1(f"[VALIDATOR] Skip label '{label}' @ frac={frac:.2f} – too few rows ({Z.shape[0]})")
                return

            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            y_cpu = cp.asnumpy(y)
            rf_scores = []
            t0 = time.time()

            for train_idx, test_idx in skf.split(cp.zeros(len(y_cpu)), y_cpu):
                Z_train, Z_test = Z[train_idx], Z[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                with _GPU_LOCK:
                    rf = RandomForestClassifier(
                        n_estimators=25,      # ↓ from 50
                        max_depth=8,          # ↓ from 10
                        max_features='sqrt',  # usually good enough
                        n_bins=64,            # control memory usage
                        accuracy_metric='balanced_accuracy'
                    )
                    rf.fit(Z_train, y_train)
                    y_pred_rf = rf.predict(Z_test)
                    cp.cuda.runtime.deviceSynchronize()

                rf_acc = float(accuracy_score(cp.asnumpy(y_test), cp.asnumpy(y_pred_rf)))
                rf_scores.append(rf_acc)

            elapsed = time.time() - t0
            job.stats.setdefault("validation", {}).setdefault(label, {}).setdefault(frac, {})["rf_time"] = elapsed

            mean_rf = round(np.mean(rf_scores), 4)
            logging.debug1(f"[RF] label='{label}' frac={frac:.2f} RF={mean_rf:.4f}")

            job.attrs.results_cupy.setdefault(label, {}).setdefault(frac, {})
            job.attrs.results_cupy[label][frac]['rf_acc'] = mean_rf




           



    # ------------------------------------------------------------------
    # Internal helpers – clustering
    # ------------------------------------------------------------------
    @staticmethod
    def _hdbscan_and_store(job: ModelerJob, Z: cp.ndarray, y: cp.ndarray, label: str, frac: float) -> None:
        """Run HDBSCAN, compute ARI/NMI/Silhouette, store metrics."""
        try:
            t0 = time.time()

            # -------- pre‑flight sanitization --------
            with _GPU_LOCK:
                Z_gpu: cp.ndarray = cp.ascontiguousarray(Z.copy())
                if Z_gpu.dtype != cp.float32:
                    Z_gpu = Z_gpu.astype(cp.float32)

                has_nan = cp.isnan(Z_gpu).any()
                has_inf = not cp.isfinite(Z_gpu).all()
                constant_cols = int(cp.any(cp.std(Z_gpu, axis=0) == 0))
                zero_dim = Z_gpu.shape[1] == 0
                too_small = Z_gpu.size == 0 or Z_gpu.shape[0] < 50

                if too_small:
                    logging.debug1(f"[HDBSCAN] Skip label='{label}' frac={frac:.2f} – too few rows")
                    return
                if has_nan:
                    logging.debug1(f"[HDBSCAN] Skip label='{label}' frac={frac:.2f} – NaNs in input")
                    return
                if has_inf:
                    logging.debug1(f"[HDBSCAN] Skip label='{label}' frac={frac:.2f} – non-finite values")
                    return
                if constant_cols:
                    logging.debug1(f"[HDBSCAN] Skip label='{label}' frac={frac:.2f} – constant column(s)")
                    return
                if zero_dim:
                    logging.debug1(f"[HDBSCAN] Skip label='{label}' frac={frac:.2f} – zero-dimensional input")
                    return

                logging.debug1(
                    f"[HDBSCAN] Passed input checks for label='{label}' frac={frac:.2f} | "
                    f"rows={Z_gpu.shape[0]}, cols={Z_gpu.shape[1]}, dtype={Z_gpu.dtype}, "
                    f"has_nan={has_nan}, has_inf={has_inf}, zero_dim={zero_dim}, constant_cols={constant_cols}"
                )

                # -------- clustering --------
                min_cluster_size = max(15, int(Z_gpu.shape[0] / 50))
                logging.debug1(
                    f"[HDBSCAN] Running HDBSCAN for label='{label}' frac={frac:.2f} | "
                    f"rows={Z_gpu.shape[0]}, cols={Z_gpu.shape[1]}, dtype={Z_gpu.dtype}, "
                    f"min_cluster_size={min_cluster_size}"
                )
                try:
                    clusterer = HDBSCAN(
                        min_cluster_size=min_cluster_size,
                    )
                    cluster_labels: cp.ndarray = clusterer.fit_predict(Z_gpu)
                except Exception as hdb_exc:
                    logging.error(
                        f"[HDBSCAN] fit_predict CRASH for label='{label}' frac={frac:.2f} | "
                        f"rows={Z_gpu.shape[0]}, cols={Z_gpu.shape[1]}, dtype={Z_gpu.dtype}, "
                        f"min_cluster_size={min_cluster_size}, err={hdb_exc}"
                    )
                    return

            # -------- metrics --------
            cluster_labels_host = cp.asnumpy(cluster_labels)
            y_host = cp.asnumpy(y)
            valid_mask = cluster_labels_host >= 0
            if valid_mask.sum() < 2:
                logging.debug1(
                    f"[HDBSCAN] Skip label='{label}' frac={frac:.2f} – <2 valid cluster labels"
                )
                return

            ari = adjusted_rand_score(y_host[valid_mask], cluster_labels_host[valid_mask])
            nmi = normalized_mutual_info_score(
                y_host[valid_mask], cluster_labels_host[valid_mask]
            )
            sil = silhouette_score(cp.asnumpy(Z_gpu), cluster_labels_host)

            hdb_time = time.time() - t0

            job.stats.setdefault("validation", {}).setdefault(label, {}).setdefault(frac, {})["hdbscan_time"] = hdb_time

            logging.debug1(
                f"[HDBSCAN] label='{label}' frac={frac:.2f} ARI={ari:.4f} NMI={nmi:.4f} sil={sil:.4f}"
            )

            job.attrs.results_cupy.setdefault(label, {}).setdefault(frac, {})
            job.attrs.results_cupy[label][frac]["ari"] = ari
            job.attrs.results_cupy[label][frac]["nmi"] = nmi
            job.attrs.results_cupy[label][frac]["silhouette"] = sil

        except Exception as exc:
            logging.warning(
                f"[HDBSCAN] failed label='{label}' frac={frac:.2f}: {exc!s}"
            )


    @staticmethod
    def _hdbscan_and_store_wrapper(job: ModelerJob, Z: cp.ndarray, y: cp.ndarray, label: str, frac: float) -> None:
        """Run default and adaptive HDBSCAN with full checks."""
        # Variant 1: default min_cluster_size
        min_cluster_default = 20
        TA52_C_Validator._run_hdbscan_variant(
            job, Z, y, label, frac, min_cluster_size=min_cluster_default, variant_name="default"
        )

        # Variant 2: adaptive min_cluster_size (based on label granularity)
        scope = job.input.scope
        count = job.attrs.uniques.get(label, {}).get(frac, {}).get(f"{scope}_n_unique", 0)

        

        if count > min_cluster_default:
            min_cluster_adaptive = 0.9 * count
            TA52_C_Validator._run_hdbscan_variant(
                job, Z, y, label, frac, min_cluster_size=min_cluster_adaptive, variant_name="adaptive"
            )


    @staticmethod
    def _run_hdbscan_variant(
        job: ModelerJob,
        Z: cp.ndarray,
        y: cp.ndarray,
        label: str,
        frac: float,
        min_cluster_size: int,
        variant_name: str
    ) -> None:
        """Run a single HDBSCAN variant with all preflight checks and store metrics."""
        try:
            t0 = time.time()

            with _GPU_LOCK:
                Z_gpu: cp.ndarray = cp.ascontiguousarray(Z.copy())
                if Z_gpu.dtype != cp.float32:
                    Z_gpu = Z_gpu.astype(cp.float32)

                has_nan = cp.isnan(Z_gpu).any()
                has_inf = not cp.isfinite(Z_gpu).all()
                constant_cols = int(cp.any(cp.std(Z_gpu, axis=0) == 0))
                zero_dim = Z_gpu.shape[1] == 0
                too_small = Z_gpu.size == 0 or Z_gpu.shape[0] < 50
                too_small_cluster = min_cluster_size >= Z_gpu.shape[0]

                if too_small:
                    logging.warning(f"[HDBSCAN-{variant_name}] Skip '{label}' frac={frac:.2f} – too few rows")
                    return
                if has_nan:
                    logging.warning(f"[HDBSCAN-{variant_name}] Skip '{label}' frac={frac:.2f} – NaNs in input")
                    return
                if has_inf:
                    logging.warning(f"[HDBSCAN-{variant_name}] Skip '{label}' frac={frac:.2f} – non-finite values")
                    return
                if constant_cols:
                    logging.warning(f"[HDBSCAN-{variant_name}] Skip '{label}' frac={frac:.2f} – constant column(s)")
                    return
                if zero_dim:
                    logging.warning(f"[HDBSCAN-{variant_name}] Skip '{label}' frac={frac:.2f} – zero-dimensional input")
                    return
                if too_small_cluster:
                    logging.warning(f"[HDBSCAN-{variant_name}] Skip '{label}' frac={frac:.2f} – cluster size too large for data")
                    return

                logging.debug1(
                    f"[HDBSCAN-{variant_name}] Passed checks for '{label}' frac={frac:.2f} | "
                    f"min_cluster_size={min_cluster_size}"
                )

                clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
                cluster_labels: cp.ndarray = clusterer.fit_predict(Z_gpu)
                cp.cuda.runtime.deviceSynchronize()

                # ---- Metrics ----
                cluster_labels_host = cp.asnumpy(cluster_labels)
                y_host = cp.asnumpy(y)
                valid_mask = cluster_labels_host >= 0

                if valid_mask.sum() < 2:
                    logging.debug1(f"[HDBSCAN-{variant_name}] <2 valid clusters for '{label}' frac={frac:.2f}")
                    return

                ari = adjusted_rand_score(y_host[valid_mask], cluster_labels_host[valid_mask])
                nmi = normalized_mutual_info_score(y_host[valid_mask], cluster_labels_host[valid_mask])
                sil = silhouette_score(cp.asnumpy(Z_gpu), cluster_labels_host)

            job.attrs.results_cupy.setdefault(label, {}).setdefault(frac, {}).update({
                f"ari_{variant_name}": ari,
                f"nmi_{variant_name}": nmi,
                f"silhouette_{variant_name}": sil,
            })

            job.stats.setdefault("validation", {}).setdefault(label, {}).setdefault(frac, {})[
                f"hdbscan_time_{variant_name}"
            ] = time.time() - t0

            logging.debug1(
                f"[HDBSCAN-{variant_name}] '{label}' frac={frac:.2f} | ARI={ari:.4f}, NMI={nmi:.4f}, Sil={sil:.4f}"
            )

        except Exception as exc:
            logging.warning(f"[HDBSCAN-{variant_name}] failed '{label}' frac={frac:.2f}: {exc}")


    # ------------------------------------------------------------------
    # Persistence helpers – results table
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_uniques_for_scope(job: ModelerJob, y: cp.ndarray, label: str) -> Dict[str, int | None]:
        """
        Return a dict with n_unique values per index column, limited to the active scope.
        Lower-level entries (below the current scope) are set to None.
        """
        uniques = {}
        encoder = job.attrs.encoder
        data = job.attrs.data_num
        scope = getattr(job.input, "scope", None)
        scope_idx = INDEX_COLS.index(scope) if scope in INDEX_COLS else len(INDEX_COLS)

        for col in INDEX_COLS:
            idx = encoder.cols.get(col)
            if idx is not None and data is not None and idx < data.shape[1]:
                if INDEX_COLS.index(col) <= scope_idx:
                    n = int(cp.unique(data[:, idx]).shape[0])
                    uniques[f"{col}_n_unique"] = n
                else:
                    uniques[f"{col}_n_unique"] = None
            else:
                uniques[f"{col}_n_unique"] = None

        uniques["n_rows"] = int(y.shape[0])
        return uniques

    
    
    
    @staticmethod
    def store_job_result(job: ModelerJob) -> None:
        """Convert nested `results_cupy` into a flat DataFrame and persist."""
        records: List[Dict[str, Any]] = []
        scope = getattr(job.input, "scope", "default_scope")
        perf_dict: Dict[str, Dict[float, Dict[str, float]]] = (
            job.attrs.results_cupy if job.attrs.results_cupy is not None else {}
        )


        for label, frac_metrics in perf_dict.items():
            for frac, metrics in frac_metrics.items():
                uniques_dict = (
                    job.attrs.uniques.get(label, {}).get(frac, {}) if job.attrs.uniques else {}
                )

                row = {
                    "DoE_UUID": job.parent_job_uuids[0],
                    "scope": scope,
                    "frac": frac,
                    "label": label,
                    "randomforest_acc": metrics.get("rf_acc"),
                    "knn_acc": metrics.get("knn_acc"),
                    "hdbscan_ari_default": metrics.get("ari_default"),
                    "hdbscan_ari_adaptive": metrics.get("ari_adaptive"),
                    "hdbscan_nmi_default": metrics.get("nmi_default"),
                    "hdbscan_nmi_adaptive": metrics.get("nmi_adaptive"),
                    "hdbscan_silhouette_default": metrics.get("silhouette_default"),
                    "hdbscan_silhouette_adaptive": metrics.get("silhouette_adaptive"),
           
                }

                row.update(uniques_dict)  # merge in unique counts
                records.append(row)


        if not records:
            logging.debug2(f"[RESULTS] No metrics to store for job {job.job_uuid}")
            job.status = "FAILED"
            return

        df = pd.DataFrame.from_records(records)
        return df
       


    # ------------------------------------------------------------------
    # Persistence helpers – UMAP cluster map
    # ------------------------------------------------------------------
    @staticmethod
    def store_ClusterEmbeddings(job: ModelerJob) -> Optional[pd.DataFrame]:
        """
        Compute and return 2D UMAP embeddings for each label within a scope, using the best
        scoring model with frac < 0.25. Annotates each point with its hierarchy labels.

        Returns
        -------
        pd.DataFrame or None
        """
        df = job.attrs.result_df
        if df is None or df.empty:
            logging.debug2(f"[UMAP] No results DataFrame for job {job.job_uuid}")
            return None

        rows = []
        data_num = job.attrs.data_num
        encoder = job.attrs.encoder.cols
        pca_results = job.attrs.multi_pca_results

        grouped = (
            df[df["frac"] < 0.25]
            .sort_values(["label", "scope", "frac"])
            .groupby(["scope", "label"])
        )

        for (scope, label), group in grouped:
            best_row = group.iloc[0]
            frac = best_row["frac"]

            Z = pca_results.get(frac, {}).get("Z_total")
            if Z is None or Z.shape[0] < 3:
                continue

            try:
                with _GPU_LOCK:
                    Z_gpu = cp.ascontiguousarray(Z)
                    if Z_gpu.dtype != cp.float32:
                        Z_gpu = Z_gpu.astype(cp.float32)

                    if Z_gpu.shape[1] > 50:
                        Z_reduced = PCA(n_components=50).fit_transform(Z_gpu)
                    else:
                        Z_reduced = Z_gpu

                    Z_umap = UMAP(n_components=2, random_state=42).fit_transform(Z_reduced)
                    Z_umap_host = cp.asnumpy(Z_umap)
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.cuda.runtime.deviceSynchronize()
            except Exception as e:
                logging.warning(f"[UMAP] Failed embedding {scope}/{label} frac={frac:.2f}: {e}")
                continue

            # determine hierarchy
            scope_idx = INDEX_COLS.index(scope) if scope in INDEX_COLS else -1
            active_levels = INDEX_COLS[: scope_idx + 1]

            for i, (x_val, y_val) in enumerate(Z_umap_host):
                row = {
                    "DoE_UUID": job.job_uuid,
                    "frac": frac,
                    "scope": scope,
                    "label": label,
                    "x": float(x_val),
                    "y": float(y_val),
                }
                for col in active_levels:
                    idx = encoder.get(col)
                    if idx is not None:
                        row[col] = int(data_num[i, idx])
                    else:
                        row[col] = None
                rows.append(row)

        if not rows:
            logging.debug2(f"[UMAP] No embeddings extracted for job {job.job_uuid}")
            return None

        df_result = pd.DataFrame(rows)

        required_cols = [
        "family", "genus", "species", "sourceID",
        "specimenID", "sampleID", "stackID", "shotID"
        ]

        # Ensure columns exist (fill missing ones with None)
        for col in required_cols:
            if col not in df_result.columns:
                df_result[col] = None

        
        return df_result

    # ------------------------------------------------------------------
    # Helper – aggregate log
    # ------------------------------------------------------------------
    @staticmethod
    def _log_summary(job: ModelerJob) -> None:
        """Store summary stats in job.stats['validation_summary'] for orchestrator-level aggregation."""
        timings = TA52_C_Validator.extract_timings_from_stats(job)

        # Predefine everything to avoid UnboundLocalError
        rf_scores, knn_scores = [], []
        ari_default, ari_adaptive = [], []
        nmi_default, nmi_adaptive = [], []
        sil_default, sil_adaptive = [], []

        df = job.attrs.result_df
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            rf_scores = df["randomforest_acc"].dropna().tolist() if "randomforest_acc" in df.columns else []
            knn_scores = df["knn_acc"].dropna().tolist() if "knn_acc" in df.columns else []

            ari_default = df["hdbscan_ari_default"].dropna().tolist() if "hdbscan_ari_default" in df.columns else []
            ari_adaptive = df["hdbscan_ari_adaptive"].dropna().tolist() if "hdbscan_ari_adaptive" in df.columns else []

            nmi_default = df["hdbscan_nmi_default"].dropna().tolist() if "hdbscan_nmi_default" in df.columns else []
            nmi_adaptive = df["hdbscan_nmi_adaptive"].dropna().tolist() if "hdbscan_nmi_adaptive" in df.columns else []

            sil_default = df["hdbscan_silhouette_default"].dropna().tolist() if "hdbscan_silhouette_default" in df.columns else []
            sil_adaptive = df["hdbscan_silhouette_adaptive"].dropna().tolist() if "hdbscan_silhouette_adaptive" in df.columns else []

        def stats(vals: List[float]) -> Tuple[float | None, float | None, float | None]:
            if not vals:
                return None, None, None
            return float(np.min(vals)), float(np.max(vals)), float(np.mean(vals))

        _, _, rf_avg = stats(rf_scores)
        _, _, knn_avg = stats(knn_scores)
        _, _, ari_avg_d = stats(ari_default)
        _, _, ari_avg_a = stats(ari_adaptive)
        _, _, nmi_avg_d = stats(nmi_default)
        _, _, nmi_avg_a = stats(nmi_adaptive)
        _, _, sil_avg_d = stats(sil_default)
        _, _, sil_avg_a = stats(sil_adaptive)

        job.stats["validation"] = {
            "rf_time": timings.get("RandomForest", 0.0),
            "knn_time": timings.get("KNN", 0.0),
            "hdbscan_time": timings.get("HDBSCAN", 0.0),
            "rf_avg": rf_avg,
            "knn_avg": knn_avg,
            "ari_avg_default": ari_avg_d,
            "ari_avg_adaptive": ari_avg_a,
            "nmi_avg_default": nmi_avg_d,
            "nmi_avg_adaptive": nmi_avg_a,
            "sil_avg_default": sil_avg_d,
            "sil_avg_adaptive": sil_avg_a,
            "total_s": sum(timings.values()),
        }





    @staticmethod
    def extract_timings_from_stats(job: ModelerJob) -> Dict[str, float]:
        """Flatten validation timings from job.stats into a summary dictionary."""
        timings = {"RandomForest": 0.0, "KNN": 0.0, "HDBSCAN": 0.0}
        validation = job.stats.get("validation", {})

        for label, frac_dict in validation.items():
            for frac, time_metrics in frac_dict.items():
                timings["RandomForest"] += time_metrics.get("rf_time", 0.0)
                timings["KNN"] += time_metrics.get("knn_time", 0.0)
                timings["HDBSCAN"] += time_metrics.get("hdbscan_time", 0.0)

        return timings
