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

    index_cols_train = bin_dict.get("index", {}).get("X")
    index_cols_test = job.attrs.data_test[:, :index_cols_train.shape[1]]
    
    if not cp.all(cp.floor(index_cols_test) == index_cols_test):
        message = "[CHECK] index_cols_test contains non-integer values!"

        logging.warning(message)

        job.status = "FAILED"

        job.input.fail_trail.mark(
            section="modelling",
            step_name="multi_PCA_index_check",
            status="error",
            error=message
        )

        return job


    if not cp.all(cp.floor(index_cols_train) == index_cols_train):
        message = "[CHECK] index_cols_train contains non-integer values!"
        logging.warning(message)

        job.status = "FAILED"

        job.input.fail_trail.mark(
            section="modelling",
            step_name="multi_PCA_index_check",
            status="error",
            error=message
        )

        return job

    index_info = bin_dict.get("index")

    blacklist = set(getattr(job.attrs, "feature_blacklist", []))
    bootstrap_No = getattr(job.input, "bootstrap_iteration", 0)

    for frac in FRACTIONS:
        logging.debug1(f"[PCA-SWEEP] Starting sweep for fraction={frac}")
        frac_start = time.time()

        outputs, col_map = [], {}
        start_idx = 0
        sweep_success = True

        outputs_train = []
        outputs_test = []

        for lbl, info in bin_dict.items():
            if lbl == "index":
                continue

            if job.attrs.data_test.shape[1] != job.attrs.data_train.shape[1]:
                logging.error(f"[PCA-SWEEP] Mismatched train/test shapes for bin '{lbl}': "
                                f"train={job.attrs.data_train.shape}, test={job.attrs.data_test.shape}")
                continue

            X_full = info["X"]
            input_cols_full = info["input_cols"]

            try:
                keep_indices = [i for i in range(X_full.shape[1]) if i not in blacklist]
                if not keep_indices:
                    raise ValueError(f"All features in bin '{lbl}' are blacklisted.")

                X_sub = X_full[:, keep_indices]
                input_cols = [input_cols_full[i] for i in keep_indices]

                # Prepare matching slice from test data:
                col_indices_test = [job.attrs.encoder.cols[c] for c in input_cols_full]
                X_test_full = job.attrs.data_test[:, col_indices_test]
                X_test_sub = X_test_full[:, keep_indices]

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
                    pca = cuPCA(n_components=n_components, output_type="cupy", whiten=pca_cfg.whiten)
                    logging.debug2_status(f"[MULTI-PCA] Starting fit-transfrom-PCA for {lbl} at fraction {frac:.2f}", overwrite=True)
                    Z_train = pca.fit_transform(X_sub)
                    logging.debug2_status(f"[MULTI-PCA] Starting fit-PCA for {lbl} at fraction {frac:.2f}", overwrite=True)

                    Z_test  = pca.transform(X_test_sub)
                    logging.debug2_status(f"[MULTI-PCA] Finished PCA for {lbl} at fraction {frac:.2f}", overwrite=True)
                except Exception:
                    logging.debug2(f"[PCA] cuML PCA failed for bin '{lbl}' @ frac={frac}. Falling back to scikit-learn PCA.")
                    pca = skPCA(n_components=n_components, whiten=pca_cfg.whiten)
                    Z_train = cp.asarray(pca.fit_transform(cp.asnumpy(X_sub)))
                    Z_test  = cp.asarray(pca.transform(cp.asnumpy(X_test_sub)))
                    logging.debug2(f"[PCA] CPU PCA completed for bin '{lbl}' @ frac={frac} → Z_train.shape={Z_train.shape}, Z_test.shape={Z_test.shape}")

                out_cols = [f"{lbl}_PC{frac:.2f}_{i+1}" for i in range(Z_train.shape[1])]
                end_idx = start_idx + Z_train.shape[1]

                col_map[lbl] = {
                    "input": input_cols,
                    "output": out_cols,
                    "dropped": [],
                    "start_idx": start_idx,
                    "end_idx": end_idx
                }

                outputs_train.append(Z_train)
                outputs_test.append(Z_test)
                start_idx = end_idx

            except Exception as e:
                sweep_success = False
                err_msg = f"[PCA] Sweep failed for bin '{lbl}' @ frac={frac}: {type(e).__name__}: {str(e)}"
                logging.warning(err_msg)
                logging.debug(traceback.format_exc())
                fail_trail_pca_errors.append((lbl, frac, err_msg))
                break  # Stop sweeping this fraction

        ## BEGINNING of the Fraq loop ---------------------------------------
        try:


            duration = time.time() - frac_start

            if not sweep_success:
                per_frac_stats[frac] = {"success": False, "duration_s": duration}
                failure_count += 1
                continue

            if index_cols_train is not None:
                index_cols = index_info["input_cols"]

                # Prepend to train output:
                outputs_train.insert(0, index_cols_train)
                outputs_test.insert(0, index_cols_test)

                # Adjust col_map:
                col_map = {
                    "index": {
                        "input": index_cols,
                        "output": index_cols,
                        "dropped": [],
                        "start_idx": 0,
                        "end_idx": len(index_cols)
                    },
                    **{
                        k: {
                            **v,
                            "start_idx": v["start_idx"] + len(index_cols),
                            "end_idx": v["end_idx"] + len(index_cols)
                        } for k, v in col_map.items()
                    }
                }
            
            fold_no = getattr(job.input, "outer_fold", 0)  # fallback to 0 if not present
            bootstrap_No = getattr(job.input, "bootstrap_iteration", 0)

            Z_total_train = cp.column_stack(outputs_train)
            Z_total_test  = cp.column_stack(outputs_test)


            job.attrs.dim_red_dict \
                .setdefault(fold_no, {}) \
                .setdefault(bootstrap_No, {})[frac] = {
                    "Z_train": Z_total_train,
                    "Z_test": Z_total_test,
                    "col_map": col_map
                }

            per_frac_stats[frac] = {"success": True, "duration_s": duration, "shape_train": Z_total_train.shape, "shape_test": Z_total_test.shape}
            success_count += 1

            logging.debug1(f"[PCA-SWEEP] Finished frac={frac:.3f} → shape_train={Z_total_train.shape} | shape_test={Z_total_test.shape} | duration={duration:.2f}s")

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

def apply_bin_dict_to_test_data(job):
    """
    Applies `bin_dict` to `job.attrs.data_test` to produce equivalent sliced test arrays.
    Returns a dict with the same keys as `bin_dict` but referencing `data_test`.
    """
    X_test = job.attrs.data_test
    bin_dict = job.attrs.bin_dict

    test_bins = {}

    for label, info in bin_dict.items():
        input_cols = info["input_cols"]
        idx = [job.attrs.encoder.cols[c] for c in input_cols]
        test_bins[label] = X_test[:, idx]

    return test_bins
