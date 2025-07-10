#  TA52_B_Modeler.py  ────────────────────────────────────────────────────────

import os
os.environ["CUML_LOG_LEVEL"] = "error"
import cuml

import time, re, logging, cupy as cp, numpy as np
from cuml.decomposition import PCA as cuPCA
from sklearn.decomposition import PCA as skPCA
from cuml.manifold import UMAP as cuUMAP
from cuml.manifold import TSNE as cuTSNE
from sklearn.manifold import TSNE as skTSNE
import umap                            # cpu fallback




FRACTIONS = [0.01, 0.02, 0.025, 0.03, 0.035, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]


class TA52_B_Modeler:
    # ─────────────────────────────────────────────────────────── public ────
    @staticmethod
    def run(job):
        logging.debug5("[MODELER] Starting feature engineering...")
        

        def step_A_check_config(job):
            """
            Step A – Validates DR config and logs method + parameters.
            """
            t0 = time.time()
            try:
                cfg = job.input.metricModel_instructions
                dr_cfg = cfg.dim_reduction_cfg

                method = dr_cfg.method                              # e.g., "t-SNE", "PCA"
                method_key = method.replace("-", "_")               # matches model field name
                method_cfg = getattr(dr_cfg, method_key, None)

                if method_cfg is None:
                    raise ValueError(f"Missing config block for method: {method}")

                # Build logging context
                job.context = f"DimReduction → {method}"
                if hasattr(method_cfg, "__dict__"):
                    job.context += f" | Params: {method_cfg.__dict__}"

                logging.debug2(job.context)

                # Store method stats
                job.stats.setdefault("modelling", {})
                job.stats["modelling"].update({
                    "method": method,
                    "config_check_s": round(time.time() - t0, 4),
                })

                # Mark success
                job.input.fail_trail.mark("modelling", "check_config", "passed")
                return job

            except Exception as e:
                logging.error(f"[MODELER][CONFIG] Failed to parse method config: {e}", exc_info=True)

                job.status = "FAILED"
                job.input.fail_trail.mark("modelling", "check_config", f"failed: {e}")

                job.stats.setdefault("modelling", {})
                job.stats["modelling"].update({
                    "method": None,
                    "config_check_s": round(time.time() - t0, 4),
                })

                return job

        job = step_A_check_config(job)
        if job.status == "FAILED":
            return job
        

        # step_01_binning.py




        def step_01_binning(job):
            """
            Step 01 – Applies binning to the job's feature matrix.

            This wrapper:
            - Measures execution time
            - Handles fail trail and job.status
            - Calls run_binning_logic() and attaches result to job.attrs.bin_dict
            - Stores binning summary in job.stats["modelling"]["bin_summary"]

            Final structure:
            job.stats["modelling"]["bin_summary"] = {
                "strategy": "implicit",
                "geo_shape": {
                    "n_cols": 3,
                    "shape": [1000, 3],
                    "col_names": [...]
                },
                ...
            }

            Returns:
                job: Updated job with bin_dict + binning stats
            """

            from app.tasks.TA52_Modeler.utils.TA52_B_utils.binning import binning
            t0 = time.time()
            job.stats.setdefault("modelling", {})
            job.stats["modelling"].setdefault("binning", {})
            binning_stats = job.stats["modelling"]["binning"]

            try:
                job = binning(job)
                bin_cfg = job.input.metricModel_instructions.binning_cfg
                strategy = bin_cfg.strategy

                # strategy at the top level
                binning_stats["strategy"] = strategy

                # per-bin entries
                for label, data in job.attrs.bin_dict.items():
                    binning_stats[label] = {
                        "n_cols": len(data["input_cols"]),
                        "shape": list(data["X"].shape),
                        "col_names": data["input_cols"]
                    }

                # timing
                job.stats["modelling"]["binning_s"] = round(time.time() - t0, 4)

                job.input.fail_trail.mark("modelling", "binning", "passed")
                return job

            except Exception as e:
                job.status = "FAILED"
                job.input.fail_trail.mark("modelling", "binning", f"failed: {e}")
                job.stats["modelling"]["binning_s"] = round(time.time() - t0, 4)
                return job

        
        job = step_01_binning(job)
        if job.status == "FAILED":
            return job


        def step_02_dim_reduction(job):
            """
            Step 02 – Applies dimensionality reduction to binned feature matrix.

            Branches based on method from job.input.metricModel_instructions.dim_reduction_cfg:
            - PCA → run_pca()
            - t-SNE / UMAP → initial PCA to threshold dims, then non-linear method
            - Autoencoder → pass all input dims

            Stores:
            - job.attrs.X_dr: Reduced feature matrix
            - job.stats["modelling"]["dim_reduction"]: result metadata
            - job.stats["modelling"]["dim_reduction_s"]: timing
            """
            from app.tasks.TA52_Modeler.utils.TA52_B_utils.reducer.multi_PCA import multi_PCA


            t0 = time.time()
            job.stats.setdefault("modelling", {})
            method = job.input.metricModel_instructions.dim_reduction_cfg.method

            try:
                match method:
                    case "PCA":
                        job = multi_PCA(job)
                    case "t-SNE":
                        raise NotImplementedError("t-SNE is not yet implemented in this step.")
                    
                        job = run_prep_PCA(job)  # initial PCA to reduce dims
                        job = run_tsne(job)
                    case "UMAP":
                        raise NotImplementedError("UMAP is not yet implemented in this step.")

                        job = run_prep_PCA(job)  # initial PCA to reduce dims
                        job = run_umap(job)
                    case "AutoencoderSeg":
                        raise NotImplementedError("AutoencoderSeg is not yet implemented in this step.")
                        pass
                    case "AutoencoderImg":
                        raise NotImplementedError("AutoencoderImg is not yet implemented in this step.")
                        pass
                    case _:
                        raise ValueError(f"Unsupported DR method: {method}")

                job.input.fail_trail.mark("modelling", "dim_reduction", "passed")
                job.stats["modelling"]["dim_reduction_s"] = round(time.time() - t0, 4)
                return job


            except Exception as e:
                job.status = "FAILED"
                job.input.fail_trail.mark("modelling", "dim_reduction", f"failed: {e}")
                job.stats["modelling"]["dim_reduction_s"] = round(time.time() - t0, 4)
                return job

        job = step_02_dim_reduction(job)
        if job.status == "FAILED":
            return job

        def step_B_wrapup(job):
            """
            Finalizes the modelling step by aggregating dimensionality reduction statistics.

            This includes:
            - Counting number of features after reduction across all tested fractions
            - Computing min, max, median, and average feature counts
            - Determining success and failure fractions
            - Storing method, bootstrap number, durations, and success rates
            - All results are saved to job.stats["modelling_summary"]

            Works generically for any dim reduction method, assuming:
            - job.attrs.dim_red_dict[frac][bootstrap] contains { "Z": ..., "col_map": ... }
            - job.input.metricModel_instructions.dim_reduction_cfg.method is defined

            Parameters
            ----------
            job : Job
                The current job containing PCA or other dim-reduction outputs and stats

            Returns
            -------
            job : Job
                The same job object with job.stats["modelling_summary"] attached
            """
            try:
                dim_red_dict = job.attrs.dim_red_dict
                method = getattr(job.input.metricModel_instructions.dim_reduction_cfg, "method", "unknown")
                bootstrap_no = getattr(job.input, "bootstrap_iteration", None)

                col_counts = []
                fractions_tested = []
                fractions_successful = []
                failures = []

                for frac, subdict in dim_red_dict.items():
                    if isinstance(subdict, dict) and bootstrap_no in subdict:
                        entry = subdict[bootstrap_no]
                        Z = entry.get("Z")
                        if Z is not None:
                            col_counts.append(Z.shape[1])
                            fractions_successful.append(frac)
                        else:
                            failures.append(frac)
                        fractions_tested.append(frac)

                col_stats_summary = {
                    "min_cols": int(np.min(col_counts)) if col_counts else 0,
                    "max_cols": int(np.max(col_counts)) if col_counts else 0,
                    "median_cols": int(np.median(col_counts)) if col_counts else 0,
                    "avg_cols": round(float(np.mean(col_counts)), 2) if col_counts else 0
                }

                avg_duration = round(
                    job.stats.get("modeler", {}).get("pca", {}).get("total_duration_s", 0)
                    / max(len(fractions_tested), 1), 3
                )

                job.stats["modelling_summary"] = {
                    "dim_reduction_method": method,
                    "success_rate": round(len(fractions_successful) / max(len(fractions_tested), 1), 2),
                    "fractions_tested": fractions_tested,
                    "fractions_successful": fractions_successful,
                    "failures": failures,
                    "duration_total_s": round(job.stats.get("modeler", {}).get("pca", {}).get("total_duration_s", 0), 3),
                    "average_duration_per_frac_s": avg_duration,
                    "n_bootstrap": bootstrap_no,
                    **col_stats_summary
                }

            except Exception as e:
                import traceback
                job.status = "FAILED"
                job.stats["modelling_summary"] = {
                    "error": f"Wrapup failed: {e}",
                    "traceback": traceback.format_exc()
                }
                import logging
                logging.error(f"[WRAPUP] Failed to compute modelling summary: {e}")
                logging.debug(traceback.format_exc())

            return job

        job = step_B_wrapup(job)
        return job




def legacy():
    pass
    # -------- multi-UMAP ---------------------------------------------------
    @staticmethod
    def _multi_umap_from_bins(bin_dict, umap_cfg):
        outputs, col_map = [], {}
        for lbl, info in bin_dict.items():
            X_sub = info["X"]
            try:
                Z = cuUMAP(output_type="cupy", **umap_cfg.dict(exclude_none=True)).fit_transform(X_sub)
            except Exception:
                Z = cp.asarray(umap.UMAP(**umap_cfg.dict(exclude_none=True))
                               .fit_transform(cp.asnumpy(X_sub)))
            out_cols = [f"{lbl}_UMAP{i+1}" for i in range(Z.shape[1])]
            outputs.append(Z)
            col_map[lbl] = {"input": info["input_cols"], "output": out_cols}
        return cp.column_stack(outputs), col_map

    # -------- multi-tSNE ---------------------------------------------------
    @staticmethod
    def _multi_tsne_from_bins(bin_dict, tsne_cfg):
        outputs, col_map = [], {}
        for lbl, info in bin_dict.items():
            X_sub = info["X"]
            try:
                Z = cuTSNE(
                    n_components=tsne_cfg.n_components,
                    perplexity=tsne_cfg.perplexity,
                    learning_rate=tsne_cfg.learning_rate,
                    n_iter=tsne_cfg.n_iter,
                    random_state=tsne_cfg.random_state,
                    method="barnes_hut",
                    init="random").fit_transform(X_sub)
            except Exception:
                Z = cp.asarray(
                        skTSNE(**tsne_cfg.dict(exclude_none=True))
                        .fit_transform(cp.asnumpy(X_sub)))
            out_cols = [f"{lbl}_TSNE{i+1}" for i in range(Z.shape[1])]
            outputs.append(Z)
            col_map[lbl] = {"input": info["input_cols"], "output": out_cols}
        return cp.column_stack(outputs), col_map

    # -------- auto-encoder placeholders -----------------------------------
    @staticmethod
    def _seg_autoencoder_from_bins(bin_dict, ae_cfg):
        raise NotImplementedError("Segmentation auto-encoder not implemented yet")

    @staticmethod
    def _img_autoencoder_from_bins(bin_dict, ae_cfg):
        raise NotImplementedError("Image auto-encoder not implemented yet")
