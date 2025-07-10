import numpy as np
import cupy as cp
import time
import traceback
import logging
from cuml.decomposition import PCA as cuPCA
from sklearn.decomposition import PCA as skPCA
from app.tasks.TA52_Modeler.TA52_B_Modeler import FRACTIONS

def multi_PCA(job):
    """
    Performs PCA sweeps per bin and component fraction, excluding blacklisted features.
    Handles bootstrapping via job.input.bootstrap_iteration and stores results in a nested dict.

    Attaches PCA results (Z + col_map) to:
        job.attrs.dim_red_dict[frac][bootstrap_iteration]

    Errors are logged per bin/fraction combo and added to job.fail_trail.modelling["multiPCA"].
    """
    bin_dict = job.attrs.bin_dict
    pca_cfg = job.input.metricModel_instructions.dim_reduction_cfg.PCA
    job.attrs.dim_red_dict = {}

    per_frac_stats = {}
    success_count, failure_count = 0, 0
    total_start = time.time()
    fail_trail_pca_errors = []

    index_X = bin_dict.get("index", {}).get("X")
    index_info = bin_dict.get("index")

    blacklist = set(getattr(job.attrs, "feature_blacklist", []))
    bootstrap_No = getattr(job.input, "bootstrap_iteration", 0)

    for frac in FRACTIONS:
        logging.debug1(f"[PCA-SWEEP] Starting sweep for fraction={frac}")
        frac_start = time.time()

        outputs, col_map = [], {}
        start_idx = 0
        sweep_success = True

        for lbl, info in bin_dict.items():
            if lbl == "index":
                continue

            X_full = info["X"]
            input_cols_full = info["input_cols"]

            try:
                keep_indices = [i for i in range(X_full.shape[1]) if i not in blacklist]
                if not keep_indices:
                    raise ValueError(f"All features in bin '{lbl}' are blacklisted.")

                X_sub = X_full[:, keep_indices]
                input_cols = [input_cols_full[i] for i in keep_indices]

                n_samples, n_features = X_sub.shape
                if n_samples < 2:
                    raise ValueError("Too few samples")
                if n_features < 2:
                    raise ValueError("Too few usable features")

                if cp.isnan(X_sub).any():
                    raise ValueError("NaNs detected before PCA")

                n_components = max(1, int(np.ceil(frac * n_features)))
                n_components = min(n_components, n_features)

                try:
                    Z = cuPCA(n_components=n_components, output_type="cupy", whiten=pca_cfg.whiten).fit_transform(X_sub)
                except Exception:
                    Z = cp.asarray(
                        skPCA(n_components=n_components, whiten=pca_cfg.whiten).fit_transform(cp.asnumpy(X_sub))
                    )

                out_cols = [f"{lbl}_PC{frac:.2f}_{i+1}" for i in range(Z.shape[1])]
                end_idx = start_idx + Z.shape[1]

                col_map[lbl] = {
                    "input": input_cols,
                    "output": out_cols,
                    "dropped": [],
                    "start_idx": start_idx,
                    "end_idx": end_idx
                }

                outputs.append(Z)
                start_idx = end_idx

            except Exception as e:
                sweep_success = False
                err_msg = f"[PCA] Sweep failed for bin '{lbl}' @ frac={frac}: {type(e).__name__}: {str(e)}"
                logging.warning(err_msg)
                logging.debug(traceback.format_exc())
                fail_trail_pca_errors.append((lbl, frac, err_msg))
                break  # Stop sweeping this fraction

        try:
            duration = time.time() - frac_start

            if not sweep_success:
                per_frac_stats[frac] = {"success": False, "duration_s": duration}
                failure_count += 1
                continue

            if index_X is not None:
                outputs.insert(0, index_X)
                col_map = {
                    "index": {
                        "input": index_info["input_cols"],
                        "output": index_info["input_cols"],
                        "dropped": [],
                        "start_idx": 0,
                        "end_idx": index_X.shape[1]
                    },
                    **{
                        k: {
                            **v,
                            "start_idx": v["start_idx"] + index_X.shape[1],
                            "end_idx": v["end_idx"] + index_X.shape[1]
                        } for k, v in col_map.items()
                    }
                }

            Z_total = cp.column_stack(outputs)

            job.attrs.dim_red_dict.setdefault(frac, {})[bootstrap_No] = {
                "Z": Z_total,
                "col_map": col_map
            }

            per_frac_stats[frac] = {"success": True, "duration_s": duration, "shape": Z_total.shape}
            success_count += 1

            logging.debug1(f"[PCA-SWEEP] Finished frac={frac:.3f} → shape={Z_total.shape} | duration={duration:.2f}s")

        except Exception as e:
            err_msg = f"[PCA] Postprocessing failed for frac={frac}: {type(e).__name__}: {str(e)}"
            logging.warning(err_msg)
            logging.debug(traceback.format_exc())
            fail_trail_pca_errors.append(("__postprocessing__", frac, err_msg))
            per_frac_stats[frac] = {"success": False, "duration_s": time.time() - frac_start}
            failure_count += 1

    # Stats
    total_duration = time.time() - total_start
    job.stats.setdefault("modelling", {})
    job.stats["modelling"].update({
        "pca": {
            "success_count": success_count,
            "failure_count": failure_count,
            "total_duration_s": total_duration,
            "per_fraction": per_frac_stats
        },
        "multiPCA": {
            "drop_col_min": 0,
            "drop_col_max": 0,
            "drop_col_avg": 0,
            "success_count": success_count,
            "failure_count": failure_count,
            "total_duration_s": total_duration
        }
    })

    # Fail trail
    if fail_trail_pca_errors:
        job.fail_trail.modelling["multiPCA"] = {
            f"{label}@frac={frac}": "failed" for (label, frac, _) in fail_trail_pca_errors
        }

    if success_count == 0:
        job.status = "FAILED"
        logging.error("[PCA-SWEEP] All PCA sweeps failed. Job marked as FAILED.")

    return job
