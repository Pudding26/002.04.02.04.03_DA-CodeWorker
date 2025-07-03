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


class TA52_B_Modeler:
    # ─────────────────────────────────────────────────────────── public ────
    @staticmethod
    def run(job):
        logging.debug5("[MODELER] Starting feature engineering...")
        X = job.attrs.data_num                       # numeric matrix from pre-processor
        cfg = job.input.metricModel_instructions
        stats = {}


        dr_cfg = job.input.metricModel_instructions.dim_reduction
        method  = dr_cfg.method          # e.g., "linear"
        family  = getattr(dr_cfg, method)  # e.g., dr_cfg.linear or dr_cfg.manifold
        submethod = family.submethod     # e.g., "PCA", "tSNE"
        subsub_cfg = getattr(family, submethod, None)  # e.g., family.PCA or family.tSNE

        job.context = f"DimReduction → {method} → {submethod}"
        if subsub_cfg:
            job.context += f" | Params: {subsub_cfg.__dict__}"
        logging.debug2(job.context)





        logging.debug2("Starting binning")
        # 1️⃣ optional BINNING → bin_dict {lbl:{X,in_cols,prefix}}
        if cfg.binning_cfg and cfg.binning_cfg.strategy != "none":
            t_beg = time.time()
            bin_dict = TA52_B_Modeler._apply_binning(job, X, cfg.binning_cfg)
            stats["binning_s"] = round(time.time() - t_beg, 4)
        else:
            bin_dict = {"_all": {"X": X, "input_cols": list(job.attrs.encoder.cols),
                                 "dest_prefix": "all"}}

        shape_before = list(X.shape)
        logging.debug2(f"[BINNING] Applied binning: {len(bin_dict)} bins created")


        # 2️⃣ DIM-REDUCTION  (per bin, then concat)
        t0 = time.time()
        Z, col_map = None, None  # Initialize variables to avoid NameError
        dr = cfg.dim_reduction
        logging.debug2(f"Starting dimensionality reduction. Method: {dr.method}")
        match dr.method:
            case "linear":
                pca_sweep = TA52_B_Modeler._multi_pca_sweep_from_bins(bin_dict, dr.linear.PCA, job)

            case "manifold":
                if dr.manifold.submethod == "UMAP":
                    Z, col_map = TA52_B_Modeler._multi_umap_from_bins(bin_dict, dr.manifold.UMAP)
                elif dr.manifold.submethod == "tSNE":
                    Z, col_map = TA52_B_Modeler._multi_tsne_from_bins(bin_dict, dr.manifold.tSNE)
                else:
                    raise ValueError(dr.manifold.submethod)
            case "encoder":
                if dr.encoder.submethod == "autoencoderSeg":
                    Z, col_map = TA52_B_Modeler._seg_autoencoder_from_bins(bin_dict, dr.encoder.AutoencoderSeg)
                elif dr.encoder.submethod == "autoencoderImg":
                    Z, col_map = TA52_B_Modeler._img_autoencoder_from_bins(bin_dict, dr.encoder.AutoencoderImg)
                else:
                    raise ValueError(dr.encoder.submethod)
            case _:
                raise ValueError(f"Unknown reduction method: {dr.method}")
        t1 = time.time()

        # 3️⃣ write-back
        if dr.method == "linear" and pca_sweep:
            logging.debug2(f"[PCA-SWEEP] Multi-PCA sweep completed with {len(pca_sweep)} fractions. Writing reults to job.")
            job.attrs.multi_pca_results = pca_sweep
            shape_after = list(pca_sweep[0.05]["Z_total"].shape)  # shape of the last bin's PCA output            

        
        if Z is not None and col_map is not None:
            job.attrs.engineered_data = Z
            job.attrs.featureClusterMap = col_map
            shape_after = list(Z.shape)

        

        # 🧠 Extract method structure for logging and grouping
        dr = cfg.dim_reduction
        method = dr.method
        family_cfg = getattr(dr, method)
        submethod = family_cfg.submethod
        subsubmethod = None
        if hasattr(family_cfg, submethod):
            sub_cfg = getattr(family_cfg, submethod)
            if sub_cfg is not None:
                # Try to find the first nested key under the submethod (like PCA, tSNE, etc.)
                subsubmethod = list(sub_cfg.model_dump(exclude_unset=True).keys())[0] if hasattr(sub_cfg, "model_dump") else None

        job.stats["feature_engineering"] = {
            "method": method,
            "submethod": submethod,
            "subsubmethod": subsubmethod,
            "shape_before":  shape_before,
            "shape_after":   shape_after,
            "shape_change":  [round(a/b,3) for a,b in zip(shape_after,shape_before)],
            "binning_s":     stats.get("binning_s", 0.0),
            "dimreduction_s":round(t1 - t0, 4),
            "total_s":       round(t1 - t0 + stats.get("binning_s",0.0), 4),
        }
        logging.debug5("[MODELER] Feature engineering completed.")

        return job

    # ──────────────────────────────────────────── helpers ──────────────────
    # central BINNING  →  returns dict
    @staticmethod
    def _apply_binning(job, X, bin_cfg):
        name2idx = job.attrs.encoder.cols  # {col_name: index}
        idx2name = {v: k for k, v in name2idx.items()}
        all_col_names = list(name2idx.keys())
        assigned_cols = set()
        out = {}

        if bin_cfg.strategy == "explicit":
            exp, presets = bin_cfg.explicit, (bin_cfg.explicit.presets or {})
            for label, members in exp.clusters.items():
                col_names = []
                for m in members:
                    col_names += presets[m] if m in presets else [m]
                idx = [name2idx[c] for c in col_names]
                bin_X = X[:, idx]
                assigned_cols.update(col_names)
                out[label] = {
                    "X": bin_X,
                    "input_cols": col_names,
                    "dest_prefix": label
                }

                logging.debug2(
                    f"[BINNING] Bin '{label}': shape={bin_X.shape}, "
                    f"cols=[{', '.join(col_names[:5])} ... {', '.join(col_names[-3:])}]"
                )

        else:  # implicit
            imp = bin_cfg.implicit
            flags = re.I if imp.ignore_case else 0
            for label, subs in imp.patterns.items():
                pat = re.compile("|".join(map(re.escape, subs)), flags)
                idx = [i for i, n in idx2name.items() if pat.search(n)]
                if not idx:
                    logging.warning(f"[BINNING] Bin '{label}' matched 0 columns.")
                    continue
                input_cols = [idx2name[i] for i in idx]
                bin_X = X[:, idx]
                assigned_cols.update(input_cols)
                out[label] = {
                    "X": bin_X,
                    "input_cols": input_cols,
                    "dest_prefix": label
                }

                logging.debug2(
                    f"[BINNING] Bin '{label}': shape={bin_X.shape}, "
                    f"cols=[{', '.join(input_cols[:5])} ... {', '.join(input_cols[-3:])}]"
                )

        index_cols = job.attrs.blacklist.get("index_cols", [])
        if index_cols:
            idx = [name2idx[c] for c in index_cols]
            bin_X = X[:, idx]
            out["index"] = {
                "X": bin_X,
                "input_cols": index_cols,
                "dest_prefix": "index"
            }

            logging.debug2(
                f"[BINNING] Bin 'index': shape={bin_X.shape}, "
                f"cols={index_cols}"
            )


        # Compute "rest" bin
        unassigned_cols = sorted(set(all_col_names) - assigned_cols)
        if unassigned_cols:
            rest_idx = [name2idx[c] for c in unassigned_cols]
            bin_X = X[:, rest_idx]
            out["rest"] = {
                "X": bin_X,
                "input_cols": unassigned_cols,
                "dest_prefix": "rest"
            }

            logging.debug2(
                f"[BINNING] Bin 'rest': shape={bin_X.shape}, "
                f"cols=[{', '.join(unassigned_cols[:5])} ... {', '.join(unassigned_cols[-3:])}]"
            )

        return out
    
    @staticmethod
    def drop_globally_constant_features(job, ignore_indices=[]):
        """
        Drops features (columns) in job.attrs.data_num that have exactly zero variance across all samples,
        excluding any feature indices in ignore_indices.

        Also updates job.attrs.colname_encoder if present to reflect the new feature set.

        Parameters:
            job: object
                Job-like object with attrs.data_num (CuPy ndarray) and optionally colname_encoder.
            ignore_indices: list[int]
                Column indices to skip when dropping features (e.g., ID columns, fraction weights).
        """
        X = job.attrs.data_num
        if X is None:
            raise ValueError("job.attrs.data_num is None")

        # Compute variance for each feature across all samples
        var = cp.var(X, axis=0)

        # Build a mask to retain features: True if variance > 0 or index is protected
        full_mask = cp.ones(X.shape[1], dtype=bool)
        for i in range(X.shape[1]):
            if i not in ignore_indices:
                full_mask[i] = var[i] > 0

        # Safety check: ensure at least one feature remains
        if not cp.any(full_mask):
            raise ValueError("No features left after global constant-feature filtering.")

        # Apply the feature mask
        job.attrs.data_num = X[:, full_mask]

        # Update column name encoder to match new feature set
        if hasattr(job.attrs, "colname_encoder") and job.attrs.colname_encoder is not None:
            job.attrs.colname_encoder = [
                name for i, name in enumerate(job.attrs.colname_encoder) if full_mask[i]
            ]

        # Log dropped feature count
        dropped = int((~full_mask).sum().item())
        if dropped > 0:
            logging.warning(
                f"Dropped {dropped} globally constant features (zero variance across all samples)."
            )


    @staticmethod
    def sanitize_X_sub(X_sub, colname_encoder=None, threshold=1e-8):
        """
        Cleans a bin-level feature matrix by:
        - Replacing NaNs and Infs
        - Identifying and dropping low-variance columns
        Returns both cleaned matrix and list of dropped column names.

        Parameters:
            X_sub (cp.ndarray): bin-local feature matrix
            colname_encoder (list[str]): names of the columns
            threshold (float): variance threshold

        Returns:
            X_filtered (cp.ndarray): cleaned matrix
            dropped_cols (list[str]): names of dropped columns
        """
        if not isinstance(X_sub, cp.ndarray):
            raise TypeError("X_sub must be a CuPy array")

        # Stabilize numerics
        X_sub = cp.nan_to_num(X_sub, nan=0.0, posinf=1e6, neginf=-1e6)

        # Calculate variance mask
        var = cp.var(X_sub, axis=0)
        keep_mask = var > threshold

        if not cp.any(keep_mask):
            raise ValueError("All features were below variance threshold in this bin.")

        # Filter X
        X_filtered = X_sub[:, keep_mask]

        # Derive dropped column names
        dropped_cols = []
        if colname_encoder:
            dropped_cols = [name for i, name in enumerate(colname_encoder) if not keep_mask[i]]

        return X_filtered, dropped_cols







    # -------- multi-PCA ----------------------------------------------------
    @staticmethod
    def _multi_pca_sweep_from_bins(bin_dict, pca_cfg, job=None):
        """
        Perform PCA sweeps over multiple fractions of components for all feature bins.

        Applies per-bin feature sanitation (drops low-variance cols),
        handles cuPCA/skPCA fallback, and builds a col_map with traceable dropped features.

        Parameters
        ----------
        bin_dict : dict
            Feature bins with CuPy arrays and metadata.
        pca_cfg : config object
            Holds PCA settings (e.g., `whiten`).
        job : Job (optional)
            If provided, stats will be stored and job.status may be set to FAILED.

        Returns
        -------
        sweep_results : dict
            Map of fraction -> dict { Z_total, col_map, success, duration_s }
        """
        import numpy as np
        import cupy as cp
        import logging
        import time
        import traceback
        from cuml.decomposition import PCA as cuPCA
        from sklearn.decomposition import PCA as skPCA

        logging.debug3("[PCA-SWEEP] Starting multi-PCA sweep...")
        fractions = [0.01, 0.02, 0.025, 0.03, 0.035, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
        sweep_results = {}

        success_count = 0
        failure_count = 0
        per_frac_stats = {}
        total_start = time.time()

        index_X = None
        index_info = None
        if "index" in bin_dict:
            index_info = bin_dict["index"]
            index_X = index_info["X"]
            logging.debug1(f"[PCA-SWEEP] Index bin detected → shape={index_X.shape}")

        for frac in fractions:
            logging.debug1(f"[PCA-SWEEP] Starting sweep for fraction={frac}")
            frac_start = time.time()

            outputs = []
            col_map = {}
            start_idx = 0
            sweep_success = True

            for lbl, info in bin_dict.items():
                if lbl == "index":
                    continue

                X_sub_orig = info["X"]
                input_cols_orig = info["input_cols"]

                try:
                    n_samples, _ = X_sub_orig.shape
                    if n_samples < 2:
                        raise ValueError(f"Too few samples for PCA: shape={X_sub_orig.shape}")
                    if cp.std(X_sub_orig) == 0:
                        raise ValueError("Input has zero variance")

                    X_sub, dropped_cols = TA52_B_Modeler.sanitize_X_sub(X_sub_orig, input_cols_orig)
                    input_cols = [col for col in input_cols_orig if col not in dropped_cols]
                    n_features_clean = X_sub.shape[1]

                    if dropped_cols:
                        logging.debug1(f"[PCA] Bin='{lbl}' dropped {len(dropped_cols)} low-variance cols")

                    if n_features_clean < 2:
                        raise ValueError(f"Too few usable features after sanitation: {n_features_clean}")

                    n_components = max(1, int(np.ceil(frac * n_features_clean)))
                    if n_components > n_features_clean:
                        logging.debug2(f"[PCA] Adjusted n_components from {n_components} → {n_features_clean} for bin '{lbl}'")
                        n_components = n_features_clean

                    Z = cuPCA(n_components=n_components, output_type="cupy", whiten=pca_cfg.whiten).fit_transform(X_sub)
                    logging.debug1(f"[PCA] cuPCA succeeded for bin '{lbl}' → shape={Z.shape}")

                except Exception as e:
                    logging.debug1(f"[PCA WARNING] cuPCA failed for bin '{lbl}' @ {frac}: {e}")
                    try:
                        Z = cp.asarray(
                            skPCA(n_components=n_components, whiten=pca_cfg.whiten).fit_transform(cp.asnumpy(X_sub_orig))
                        )
                        logging.debug1(f"[PCA] skPCA fallback succeeded for bin '{lbl}'")
                        dropped_cols = []
                        input_cols = input_cols_orig
                    except Exception as e2:
                        logging.warning(f"[PCA FAILURE] skPCA also failed for bin '{lbl}' @ {frac}: {e2}")
                        logging.debug(traceback.format_exc())
                        sweep_success = False
                        break

                out_cols = [f"{lbl}_PC{frac:.2f}_{i+1}" for i in range(Z.shape[1])]
                end_idx = start_idx + Z.shape[1]

                col_map[lbl] = {
                    "input": input_cols,
                    "output": out_cols,
                    "dropped": dropped_cols,
                    "start_idx": start_idx,
                    "end_idx": end_idx
                }

                outputs.append(Z)
                start_idx = end_idx

            duration = time.time() - frac_start

            if not sweep_success:
                logging.warning(f"[PCA-SWEEP] Skipping frac={frac:.3f} due to failure. Took {duration:.2f}s.")
                sweep_results[frac] = {
                    "Z_total": None,
                    "col_map": None,
                    "success": False,
                    "duration_s": duration
                }
                failure_count += 1
                per_frac_stats[frac] = {"success": False, "duration_s": duration}
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
                        }
                        for k, v in col_map.items()
                    }
                }

            Z_total = cp.column_stack(outputs)
            logging.debug1(f"[PCA-SWEEP] Finished frac={frac:.3f} → shape={Z_total.shape} | duration={duration:.2f}s")

            sweep_results[frac] = {
                "Z_total": Z_total,
                "col_map": col_map,
                "success": True,
                "duration_s": duration
            }
            success_count += 1
            per_frac_stats[frac] = {"success": True, "duration_s": duration}

        # ⏱ Summary
        total_duration = time.time() - total_start

        # 🔢 Compute dropped col stats
        dropped_counts = []
        for sweep in sweep_results.values():
            if not sweep["success"] or sweep["col_map"] is None:
                continue
            for cm in sweep["col_map"].values():
                dropped_counts.append(len(cm.get("dropped", [])))

        if dropped_counts:
            min_dropped = min(dropped_counts)
            max_dropped = max(dropped_counts)
            avg_dropped = round(sum(dropped_counts) / len(dropped_counts), 2)
        else:
            min_dropped = max_dropped = avg_dropped = 0

        logging.debug3(
            f"[PCA-SWEEP SUMMARY] Successful={success_count}, Failed={failure_count}, Total time={total_duration:.2f}s"
            f"dropped zero-variance cols per bin [min={min_dropped}, max={max_dropped}, avg={avg_dropped}]"
        )

        if job is not None:
            modeler_stats = job.stats.setdefault("modeler", {})
            modeler_stats["pca"] = {
                "success_count": success_count,
                "failure_count": failure_count,
                "total_duration_s": total_duration,
                "per_fraction": per_frac_stats
            }
            modeler_stats["multiPCA"] = {
                "drop_col_min": min_dropped,
                "drop_col_max": max_dropped,
                "drop_col_avg": avg_dropped,
                "success_count": success_count,
                "failure_count": failure_count,
                "total_duration_s": total_duration
            }

        if success_count == 0:
            logging.error("[PCA-SWEEP] All PCA sweeps failed. Marking job as FAILED.")
            job.status = "FAILED"

        return sweep_results







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
