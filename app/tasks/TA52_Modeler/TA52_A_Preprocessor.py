from app.utils.common.app.utils.dataModels.Jobs.ModelerJob import ModelerJob


import os
os.environ["CUML_LOG_LEVEL"] = "error"
import cuml

import cudf
import time
import re
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import cupy as cp
import numpy as np
from typing import List, Dict, Any
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from cuml.neighbors import NearestNeighbors
from app.tasks.TA52_Modeler.utils.split_numeric_non_numeric_cudf import split_numeric_non_numeric_cudf


INDEX_COLS = ["family", "genus", "species", "sourceID", "specimenID", "sampleID", "stackID", "shotID"]
import time, logging
from collections import Counter

import cupy as cp
import numpy as np
import cudf

# ────────────────────────────────────────────────────────────────────────────────
class TA52_A_Preprocessor:
    """
    Unified pre-processing entry point.

    ▸ `run(job)` is unchanged except for calling the new helper signatures.
    ▸ All helpers now accept exactly one argument: the `job` object.
    ▸ Label/target extraction is done via `_split_Xy`, using `index_col`
      (or a legacy numeric `scope`) that the dataprep step already wrote
      into the config object.  No encoder look-ups, no string dataframes.
    """

    # ─────────────────────────────────────────────────────────────────── run ────
    @staticmethod
    def run(job) -> None:
        """
        Main orchestration.  Assumes `job.attrs.data_num` already exists
        and columns were encoded + the target column index recorded in
        the pre-processing config.
        """
        logging.debug3("STARTING Preprocessor")

        t0 = time.time()
        #TA52_A_Preprocessor.expand_raw_data(job, factor=500)  # <- optional synthetic expansion
        # <- call your existing prepare_numeric_gpu_data here -------------
        logging.debug2("Preparing numeric GPU data...")
        TA52_A_Preprocessor.prepare_numeric_gpu_data(job)
        logging.debug2("Numeric GPU data prepared.")
        cfg          = job.input.preProcessing_instructions
        X_gpu        = job.attrs.data_num
        shape_before = list(X_gpu.shape)
        t1           = time.time()
        job.stats.setdefault("preprocessing", {})

        if cfg.method is None:
            logging.warning("No preprocessing method specified, skipping.")
            job.attrs.preProcessed_data = X_gpu
            job.stats["preprocessing"] = {
                "method": None,
                "submethod": None,
                "subsubmethod": None,
                "shape_before": shape_before,
                "shape_after": shape_before,
                "elapsed_time": 0.0,
            }
            return job

        # ---------------------------------------------------------------- config
        try:
            method_cfg = getattr(cfg, cfg.method)
        except Exception:
            logging.error(
                f"[PREPROCESS ERROR] Failed to access sub-config for method='{cfg.method}' "
                f"(type: {type(cfg.method)}). Job ID: {getattr(job, 'id', 'unknown')}",
                exc_info=True
            )
            job.status = "FAILED"
            return job

        subsubmethod = getattr(method_cfg, method_cfg.submethod).subsubmethod
        job.context  = f"Preprocessing with method: {subsubmethod}"
        logging.debug2(job.context)

        # _____ Scaling
        t2 = time.time()
        logging.debug2("Scaling data...")
        job.attrs.data_num = TA52_A_Preprocessor._minmax_scaler(job, feature_range=(0, 1))

        # N/an handling and bin weighting

        t3 = time.time()

        logging.debug2("Handling NaNs and applying bin weighting...")
        job.attrs.data_num = TA52_A_Preprocessor._bin_fraction_weighting(job)
        # Drop rows with any NaNs after bin weighting
        
        job, dropped = TA52_A_Preprocessor.drop_nans(job)
        job.stats["preprocessing"]["dropped_nans_after_binning"] = dropped
        job = TA52_A_Preprocessor._check_dropped_fraction(job, shape_before, dropped)
        if job.status == "FAILED":
            logging.warning(f"[PREPROCESS] Job {job.job_uuid} failed due to excessive NaN rows dropped ({dropped} rows) after scaling.")
            return job


        t4 = time.time()
        logging.debug2("Data scaling and NaN handling completed.")
        logging.debug2(f"Applying sampling method: {method_cfg.submethod} with scope: {job.input.scope}")
        # ------------------------------------------------------------- dispatch
        match cfg.method:
            # ───────────── resampling
            case "resampling":
                match method_cfg.submethod:
                    case "undersampling":
                        X_out = TA52_A_Preprocessor._random_undersample(job)

                    case "oversampling":
                        match subsubmethod:
                            case "RandomOverSampler":
                                X_out = TA52_A_Preprocessor._random_oversample(job)
                            case "SMOTESampler":
                                X_out = TA52_A_Preprocessor._smote_sampler(job)
                            case _:
                                raise ValueError(f"Unsupported oversampling.subsubmethod: {subsubmethod}")

                    case "bootstrapping":
                        X_out = TA52_A_Preprocessor._bootstrap(job)

                    case "hybrid":
                        X_out = TA52_A_Preprocessor._hybrid_sampler(job)

                    case _:
                        raise ValueError(f"Unsupported resampling.submethod: {method_cfg.submethod}")

            # ───────────── unknown
            case _:
                raise ValueError(f"Unsupported preProcessing method: {cfg.method}")

        # ---------------------------------------------------------------- stats
        t5           = time.time()
        logging.debug2("Sampling completed")
        shape_after = list(X_out.shape)
        job.attrs.data_num = X_out

        job, dropped = TA52_A_Preprocessor.drop_nans(job)
        job = TA52_A_Preprocessor._check_dropped_fraction(job, shape_before, dropped)
        job.stats["preprocessing"]["dropped_nans_after_sampling"] = dropped
        if job.status == "FAILED":
            logging.warning(f"[PREPROCESS] Job {job.job_uuid} failed due to excessive NaN rows dropped ({dropped} rows) after sampling.")
            return job
        
        job.attrs.preProcessed_data = job.attrs.data_num  # Store the final preprocessed data

        job.context = f"SUCCESS of: Preprocessing with method: {subsubmethod}"
        logging.debug1("SUCCESS")

        t6 = time.time()
        shape_change = [
            round(a / b, 3) if b != 0 else None
            for a, b in zip(shape_after, shape_before)
        ]

        # ✅ Fix: use parentheses for update()
        job.stats["preprocessing"].update({
            "method": cfg.method,
            "submethod": method_cfg.submethod,
            "subsubmethod": subsubmethod,
            "shape_before": shape_before,
            "shape_after": shape_after,
            "shape_change": shape_change,
            "create_cp_arrays": round(t1 - t0, 4),
            "load_config_s": round(t2 - t1, 4),
            "scaling_s": round(t3 - t2, 4),
            "weighting_s": round(t4 - t3, 4),
            "coreProcess_s": round(t5 - t4, 4),
            "final_assignment_s": round(t6 - t5, 4),
            "total_s": round(t6 - t0, 4)
        })

        logging.debug3(f"Preprocessing completed in {job.stats['preprocessing']['total_s']} seconds")

        return job




#----------
# UTILITY
#----------

    @staticmethod
    def drop_nans(job):
        """Drop rows with any NaNs in job.attrs.data_num and apply mask to encoder_vals if present."""
        X = job.attrs.data_num
        if X is None:
            raise ValueError("job.attrs.data_num is None – cannot drop NaNs.")

        mask = ~cp.isnan(X).any(axis=1)
        dropped = int(X.shape[0] - cp.sum(mask).item())
        if dropped > 0:
            logging.warning(f"[drop_nans] Dropped {dropped} rows with NaNs.")

        # Apply mask to data
        job.attrs.data_num = X[mask]

        # Apply same mask to encoder if present
        if hasattr(job.attrs, "encoder_vals") and job.attrs.encoder_vals is not None:
            job.attrs.encoder_vals = job.attrs.encoder_vals[mask]

        return job, dropped


    @staticmethod
    def _check_dropped_fraction(job, shape_before, dropped, threshold=0.75):
        n_total = shape_before[0]
        n_remaining = job.attrs.data_num.shape[0]
        dropped_fraction = dropped / n_total if n_total > 0 else 1.0

        job.attrs.dropped_fraction = round(dropped_fraction, 4)

        if dropped_fraction > threshold:
            job.status = "FAILED"

        return job


    @staticmethod
    def _encode_columns(df: cudf.DataFrame, cols: List[str]) -> Dict[str, cudf.Series]:
        """
        In-place label-encode each column in `cols`.
        Returns a dict: {col_name: categories_series}.
        Pure GPU (cudf.Series.factorize).
        """
        encoders = {}
        for col in cols:
            codes, cats = df[col].factorize()       # GPU, O(N)
            df[col]   = codes.astype("int32")
            encoders[col] = cats                   # cats is a cudf.Series
        return encoders



    @staticmethod
    def decode_columns(df: cudf.DataFrame, encoders: dict):
        for col, cats in encoders.items():
            df[col] = cats.take(df[col].astype("int32"))
        return df


    @staticmethod
    def prepare_numeric_gpu_data(job) -> None:
        """
        1.  Load pandas DataFrame (job.input.raw_data).
        2.  Decide which hierarchy levels to keep using job.input.scope.
        3.  Encode all string / index columns on GPU.
        4.  Return:
            • job.input.data_num          (cupy.ndarray, float32)
            • job.input.encoder.cols      (dict str → int)
            • job.input.encoder.vals      (dict str → cudf.Series)
        5.  Free the original raw DataFrame to save memory.
        """
        # ------------------------------------------------------------------ #
        # 0️⃣  load → cudf
        df_raw_pd  = job.attrs.raw_data                # pandas.DataFrame
        df_raw     = cudf.from_pandas(df_raw_pd)       # GPU
        del df_raw_pd                                  # free host RAM

        # ------------------------------------------------------------------ #
        # 1️⃣  hierarchy slicing
        scope      = job.input.scope                   # e.g. "campaign"
        if scope is None:
            scope = "shotID"                           # default lowest level
        try:
            scope_idx = INDEX_COLS.index(scope)
        except ValueError:
            raise ValueError(f"scope '{scope}' not in INDEX_COLS {INDEX_COLS}")

        index_keep = INDEX_COLS[:scope_idx + 1]        # ['session', 'campaign']
        index_drop = INDEX_COLS[scope_idx + 1:]        # ['shotID']
        df_work    = df_raw.drop(columns=index_drop)   # drop finer levels

        # ------------------------------------------------------------------ #
        # 2️⃣  split columns
        num_cols   = df_work.select_dtypes(include="number").columns.tolist()

        # Index and string columns present in the current DataFrame
        index_present = [c for c in INDEX_COLS if c in df_work.columns]
        str_cols      = [c for c in df_work.columns if c not in num_cols and c not in index_present]

        # ------------------------------------------------------------------ #
        # 3️⃣  GPU label-encode
        enc_idx  = TA52_A_Preprocessor._encode_columns(df_work, index_present)
        enc_str  = TA52_A_Preprocessor._encode_columns(df_work, str_cols)
        enc_vals = {**enc_idx, **enc_str}              # merge the two dicts

        # ------------------------------------------------------------------ #
        # 4️⃣  final numeric matrix  (still cudf → now cupy ndarray)
        # Insert index + encoded str columns first, followed by numeric ones
        ordered_cols = index_present + str_cols + [c for c in num_cols if c not in index_present]
        data_num_df  = df_work[ordered_cols]

        # Replace cuDF nulls (sentinel value) with IEEE-754 NaN
        if data_num_df.isnull().any().any():
            data_num_df = data_num_df.fillna(cp.nan)

        X_gpu = data_num_df.astype("float32").values   # cupy.ndarray

        # ------------------------------------------------------------------ #
        # 5️⃣  column-name encoder (dict str → int)
        col_encoder = {col: i for i, col in enumerate(data_num_df.columns)}

        # ------------------------------------------------------------------ #
        # 6️⃣  write back into the job object
        job.attrs.data_num            = X_gpu
        job.attrs.encoder             = type("Enc", (), {})()  # tiny namespace object
        job.attrs.encoder.cols        = col_encoder
        job.attrs.encoder.vals        = enc_vals

        # 🆕 7️⃣ blacklist info
        job.attrs.blacklist = {
            "index_cols": index_present,
            "str_cols": str_cols
        }

        # 🆕 8️⃣ Store index column index in data matrix
        try:
            job.input.index_col = col_encoder[scope]
        except KeyError:
            raise ValueError(f"Could not find encoded index for scope column '{scope}' in numeric matrix.")

        # optional: drop the big cudf frame as well if memory is tight
        del df_raw, df_work, data_num_df
        cp.get_default_memory_pool().free_all_blocks()  # release GPU mem

    # ──────────────────────────────────────────────────────────── split util ────
    @staticmethod
    def _split_Xy(job, param_block=None):
        """
        Extract (X_np, y_np) using job.input.index_col for the label.
        Ignores param_block, since the correct index is globally known.
        """
        X_gpu = job.attrs.data_num
        idx   = int(job.input.index_col)
        y_np  = cp.asnumpy(X_gpu[:, idx]).astype("int32")
        X_np  = cp.asnumpy(X_gpu)
        return X_np, y_np




    @staticmethod
    def _bin_fraction_weighting(job) -> cp.ndarray:
        """
        Applies bin weighting directly using encoded column indices.
        Assumes:
            - Fraction values are already present in `job.attrs.data_num`.
            - All indices are from `job.attrs.encoder.cols`.
        """
        def extract_bin_info_from_encoder(encoder_cols):
            """
            From a flat encoder mapping (name → index), extract per-bin:
                - all feature indices
                - the single fraction column index
            Returns:
                dict: {
                    'p00-p05': {
                        'feature_indices': [...],
                        'fraction_index': int
                    }
                }
            """
            bin_info = {}

            for name, idx in encoder_cols.items():
                match = re.search(r"area_by_area_(p\d{2}-p\d{2})_", name)
                if not match:
                    continue

                bin_label = match.group(1)

                if name.endswith("_fraction"):
                    bin_info.setdefault(bin_label, {})["fraction_index"] = idx
                else:
                    bin_info.setdefault(bin_label, {}).setdefault("feature_indices", []).append(idx)

            return bin_info
        
        X = cp.asarray(job.attrs.data_num)
        bin_info = extract_bin_info_from_encoder(job.attrs.encoder.cols)

        if not bin_info:
            logging.debug1("No bin info found — skipping bin weighting.")
            return X

        X_weighted = X.copy()

        for bin_label, info in bin_info.items():
            if "fraction_index" not in info or "feature_indices" not in info:
                continue

            frac = X[:, info["fraction_index"]].reshape(-1, 1)  # (n_samples, 1)
            cols = info["feature_indices"]
            X_weighted[:, cols] = cp.nan_to_num(X[:, cols] * frac, nan=0.0)

        return X_weighted


    @staticmethod
    def expand_raw_data(job, factor: int) -> None:
        """
        Synthetically increases the size of `job.attrs.raw_data` by repeating it `factor` times.

        Args:
            job: The job object containing the original raw pandas DataFrame.
            factor: Integer multiplier (e.g., 10 means 10× the original size).
        """
        import pandas as pd

        df = job.attrs.raw_data
        if factor <= 1:
            return  # no change

        job.attrs.raw_data = pd.concat([df] * factor, ignore_index=True)
        logging.info(f"Synthetic data expansion complete: {len(df)} → {len(job.attrs.raw_data)} rows")






    # ─────────────────────────────────────────────────────────── scalers ────
    @staticmethod
    def _standard_scaler(job):
        from cuml.preprocessing import StandardScaler
        cfg = job.input.preProcessing_instructions
        p   = cfg.scaling.standardization.StandardScaler
        scaler = StandardScaler(with_mean=p.with_mean, with_std=p.with_std)
        return scaler.fit_transform(job.attrs.data_num)

    @staticmethod
    def _minmax_scaler(job, feature_range=(0, 1)):
        from cuml.preprocessing import MinMaxScaler
        cfg = job.input.preProcessing_instructions
        p   = cfg.scaling.standardization.MinMaxScaler

        X     = job.attrs.data_num
        cols  = job.attrs.encoder.cols
        bl    = job.attrs.blacklist

        # Get indices of all blacklisted columns (index + str)
        blacklist_cols = set(bl["index_cols"] + bl["str_cols"])
        blacklist_idx  = [cols[c] for c in blacklist_cols if c in cols]

        # Compute all columns to scale (those NOT in blacklist)
        scale_idx = [i for i in range(X.shape[1]) if i not in blacklist_idx]

        # Create empty array to hold scaled values
        X_scaled = cp.array(X)  # full copy, keeps unscaled values in-place

        if scale_idx:
            scaler = MinMaxScaler(feature_range=feature_range)
            X_scaled[:, scale_idx] = scaler.fit_transform(X[:, scale_idx])
        else:
            logging.warning("[SCALER] No columns left to scale after applying blacklist.")

        return X_scaled

    # ───────────────────────────────────────────────────────── samplers ────
    @staticmethod
    def _random_undersample(job):
        X = job.attrs.data_num
        idx = int(job.input.index_col)
        y = X[:, idx]

        labels, counts = cp.unique(y, return_counts=True)

        # Fetch strategy from config
        cfg = job.input.preProcessing_instructions
        p = cfg.resampling.undersampling.RandomUnderSampler
        strategy = getattr(p, "strategy", "min")

        if strategy == "min":
            target = int(cp.min(counts))
        elif strategy == "median":
            target = int(cp.median(counts))
        elif strategy == "mean":
            target = int(cp.mean(counts))
        elif isinstance(strategy, int):
            target = int(strategy)
        else:
            raise ValueError(f"Unsupported undersampling strategy: {strategy}")

        result = []
        for label in labels.tolist():
            mask = y == label
            X_class = X[mask]
            n = X_class.shape[0]
            if n > target:
                selected = cp.random.choice(n, size=target, replace=False)
                X_sampled = X_class[selected]
            else:
                X_sampled = X_class
            result.append(X_sampled)

        return cp.concatenate(result, axis=0)


    @staticmethod
    def _random_oversample(job):
        X = job.attrs.data_num
        idx = int(job.input.index_col)
        y = X[:, idx]

        labels, counts = cp.unique(y, return_counts=True)

        # Fetch strategy from config
        cfg = job.input.preProcessing_instructions
        p = cfg.resampling.oversampling.RandomOverSampler
        strategy = getattr(p, "strategy", "max")

        if strategy == "max":
            target = int(cp.max(counts))
        elif strategy == "median":
            target = int(cp.median(counts))
        elif strategy == "mean":
            target = int(cp.mean(counts))
        elif isinstance(strategy, int):
            target = int(strategy)
        else:
            raise ValueError(f"Unsupported oversampling strategy: {strategy}")

        oversampled_chunks = []

        for label in labels.tolist():
            mask = y == label
            X_c = X[mask]
            n = X_c.shape[0]

            if n < target:
                extra_idx = cp.random.choice(n, size=target - n, replace=True)
                X_extra = X_c[extra_idx]
                X_bal = cp.concatenate([X_c, X_extra], axis=0)
            else:
                X_bal = X_c

            oversampled_chunks.append(X_bal)

        return cp.concatenate(oversampled_chunks, axis=0)



    @staticmethod
    def _smote_sampler(job):
        from imblearn.over_sampling import SMOTE
        from cuml.neighbors import NearestNeighbors  # RAPIDS version for GPU
        import cupy as cp
        import numpy as np

        cfg = job.input.preProcessing_instructions
        p   = cfg.resampling.oversampling.SMOTESampler

        X_np, y_np = TA52_A_Preprocessor._split_Xy(job)

        # Filter out rows with any NaNs
        valid_mask = ~np.isnan(X_np).any(axis=1)
        X_clean = X_np[valid_mask]
        y_clean = y_np[valid_mask]

        # RAPIDS NearestNeighbors (massive GPU speedup)
        knn = NearestNeighbors(n_neighbors=p.k_neighbors, output_type='numpy')
        knn.fit(X_clean)  # fit must be called here for imblearn compatibility

        smote = SMOTE(
            k_neighbors=knn,                              # GPU-accelerated NN
            sampling_strategy=p.sampling_strategy,
            random_state=p.random_state,
        )

        X_resampled, _ = smote.fit_resample(X_clean, y_clean)

        return cp.asarray(X_resampled)


    @staticmethod
    def _bootstrap(job):
        cfg = job.input.preProcessing_instructions
        p   = cfg.resampling.bootstrapping.BootstrapSampler
        X   = job.attrs.data_num
        n   = p.n_samples or len(X)
        idx = cp.random.choice(cp.arange(len(X)), size=n, replace=True)
        return X[idx]

    @staticmethod
    def _hybrid_sampler(job):
        X = job.attrs.data_num
        idx = int(job.input.index_col)
        group_ids = X[:, idx]

        # Count group sizes
        unique_ids, counts = cp.unique(group_ids, return_counts=True)

        # Fetch strategy
        cfg = job.input.preProcessing_instructions
        p = cfg.resampling.hybrid.HybridSampler
        strategy = getattr(p, "strategy", "median")

        if strategy == "min":
            target = int(cp.min(counts))
        elif strategy == "median":
            target = int(cp.median(counts))
        elif strategy == "mean":
            target = int(cp.mean(counts))
        elif strategy == "max":
            target = int(cp.max(counts))
        elif isinstance(strategy, int):
            target = int(strategy)
        else:
            raise ValueError(f"Unsupported hybrid sampling strategy: {strategy}")

        chunks = []
        for gid in unique_ids.tolist():
            mask = group_ids == gid
            X_g = X[mask]
            n = X_g.shape[0]

            if n < target:
                extra_idx = cp.random.choice(n, size=target - n, replace=True)
                X_extra = X_g[extra_idx]
                X_bal = cp.concatenate([X_g, X_extra], axis=0)
            elif n > target:
                selected_idx = cp.random.choice(n, size=target, replace=False)
                X_bal = X_g[selected_idx]
            else:
                X_bal = X_g

            chunks.append(X_bal)

        return cp.concatenate(chunks, axis=0)


