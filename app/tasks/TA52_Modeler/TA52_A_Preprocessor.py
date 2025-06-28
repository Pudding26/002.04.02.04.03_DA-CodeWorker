from app.utils.common.app.utils.dataModels.Jobs.ModelerJob import ModelerJob
import cudf
import time
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
        t0 = time.time()
        TA52_A_Preprocessor.expand_raw_data(job, factor=1500)  # <- optional synthetic expansion
        # <- call your existing prepare_numeric_gpu_data here -------------
        TA52_A_Preprocessor.prepare_numeric_gpu_data(job)

        cfg          = job.input.preProcessing_instructions
        X_gpu        = job.attrs.data_num
        shape_before = list(X_gpu.shape)
        t1           = time.time()

        if cfg.method is None:
            logging.debug1("No preprocessing method specified, skipping.")
            job.attrs.preProcessed_data = X_gpu
            job.stats["preprocessing"] = {
                "method": None,
                "submethod": None,
                "subsubmethod": None,
                "shape_before": shape_before,
                "shape_after": shape_before,
                "elapsed_time": 0.0,
            }
            return

        # ---------------------------------------------------------------- config
        try:
            method_cfg = getattr(cfg, cfg.method)
        except Exception:
            logging.error(
                f"[PREPROCESS ERROR] Failed to access sub-config for method='{cfg.method}' "
                f"(type: {type(cfg.method)}). Job ID: {getattr(job, 'id', 'unknown')}",
                exc_info=True
            )
            raise

        subsubmethod = getattr(method_cfg, method_cfg.submethod).subsubmethod
        job.context  = f"Preprocessing with method: {subsubmethod}"
        logging.debug1(job.context)

        # ------------------------------------------------------------- dispatch
        match cfg.method:
            # ───────────── scaling
            case "scaling":
                match method_cfg.submethod:
                    case "standardization":
                        match subsubmethod:
                            case "MinMaxScaler":
                                X_out = TA52_A_Preprocessor._minmax_scaler(job)
                            case "StandardScaler":
                                X_out = TA52_A_Preprocessor._standard_scaler(job)
                            case _:
                                raise ValueError(f"Unsupported standardization.subsubmethod: {subsubmethod}")
                    case _:
                        raise ValueError(f"Unsupported scaling.submethod: {method_cfg.submethod}")

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
        t4           = time.time()
        shape_after  = list(X_out.shape)
        t5           = time.time()

        job.attrs.preProcessed_data = X_out
        job.context = f"SUCCESS of: Preprocessing with method: {subsubmethod}"
        logging.debug1("SUCCESS")
        t6 = time.time()
        shape_change = [
            round(a / b, 3) if b != 0 else None
            for a, b in zip(shape_after, shape_before)
        ]

        job.stats["preprocessing"] = {
            "method": cfg.method,
            "submethod": method_cfg.submethod,
            "subsubmethod": subsubmethod,
            "shape_before": shape_before,
            "shape_after": shape_after,
            "shape_change": shape_change,
            "split_columns_s": 0.0,       # ← placeholder (or add real timing later)
            "expand_data_s": 0.0,         # ← placeholder (or add real timing later)
            "load_config_s": round(t1 - t0, 4),
            "preprocess_core_s": round(t4 - t1, 4),
            "postprocess_s": round(t5 - t4, 4),
            "final_assignment_s": round(t6 - t5, 4),
            "total_s": round(t6 - t0, 4)
        }
        logging.debug1(f"Preprocessing completed in {job.stats['preprocessing']['total_s']} seconds")




#----------
# UTILITY
#----------

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
        str_cols   = [c for c in df_work.columns if c not in num_cols and c not in index_keep]

        # ------------------------------------------------------------------ #
        # 3️⃣  GPU label-encode
        enc_idx = TA52_A_Preprocessor._encode_columns(df_work, index_keep)
        enc_str = TA52_A_Preprocessor._encode_columns(df_work, str_cols)
        enc_vals = {**enc_idx, **enc_str}              # merge the two dicts

        # ------------------------------------------------------------------ #
        # 4️⃣  final numeric matrix  (still cudf → now cupy ndarray)
        data_num_df = df_work[num_cols + index_keep]   # keep hierarchy cols numeric
        
        # Replace cuDF nulls (sentinel value) with IEEE-754 NaN
        if data_num_df.isnull().any().any():
            data_num_df = data_num_df.fillna(cp.nan)
        
        
        X_gpu       = data_num_df.astype("float32").values   # cupy.ndarray

        # ------------------------------------------------------------------ #
        # 5️⃣  column-name encoder (dict str → int)
        col_encoder = {col: i for i, col in enumerate(data_num_df.columns)}

        # ------------------------------------------------------------------ #
        # 6️⃣  write back into the job object
        job.attrs.data_num            = X_gpu
        job.attrs.encoder             = type("Enc", (), {})()  # tiny namespace object
        job.attrs.encoder.cols        = col_encoder
        job.attrs.encoder.vals        = enc_vals

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
    def _minmax_scaler(job):
        from cuml.preprocessing import MinMaxScaler
        cfg = job.input.preProcessing_instructions
        p   = cfg.scaling.standardization.MinMaxScaler
        scaler = MinMaxScaler(feature_range=(p.min, p.max))
        return scaler.fit_transform(job.attrs.data_num)

    # ───────────────────────────────────────────────────────── samplers ────
    @staticmethod
    def _random_undersample(job):
        from imblearn.under_sampling import RandomUnderSampler
        cfg = job.input.preProcessing_instructions
        p   = cfg.resampling.undersampling.RandomUnderSampler
        X_np, y_np = TA52_A_Preprocessor._split_Xy(job, p)
        sampler = RandomUnderSampler(random_state=p.random_state)
        X_res, _ = sampler.fit_resample(X_np, y_np)
        return cp.asarray(X_res)

    @staticmethod
    def _random_oversample(job):
        cfg = job.input.preProcessing_instructions
        p   = cfg.resampling.oversampling.RandomOverSampler
        X_gpu = job.attrs.data_num
        idx   = int(job.input.index_col)


        # Work on-GPU with cudf
        df = cudf.DataFrame(X_gpu)
        df["target"] = df.iloc[:, idx]

        counts         = df["target"].value_counts()
        max_size       = counts.max()
        dfs, cache     = [], {}

        for label in counts.index.to_arrow().to_pylist():
            df_c = df[df["target"] == label]
            n    = len(df_c)
            if n < max_size:
                cache.setdefault(n, cp.arange(n))
                extra = cp.random.choice(cache[n], size=max_size - n, replace=True)
                df_c  = cudf.concat([df_c, df_c.iloc[extra]], ignore_index=True)
            dfs.append(df_c)

        over = cudf.concat(dfs, ignore_index=True).drop(columns=["target"])
        return over.to_cupy()

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
        from imblearn.over_sampling import RandomOverSampler
        from imblearn.under_sampling import RandomUnderSampler
        cfg = job.input.preProcessing_instructions
        p   = cfg.resampling.hybrid.HybridSampler
        X_np, y_np = TA52_A_Preprocessor._split_Xy(job, p)
        counts = Counter(y_np)

        # Determine target size
        if p.target_size is not None:
            target = p.target_size
        elif p.strategy == "mean":
            target = int(np.mean(list(counts.values())))
        elif p.strategy == "max":
            target = int(np.max(list(counts.values())))
        elif p.strategy == "min":
            target = int(np.min(list(counts.values())))
        else:
            target = int(np.median(list(counts.values())))

        chunks = []
        for label, size in counts.items():
            mask = y_np == label
            X_c  = X_np[mask]
            y_c  = y_np[mask]

            if size < target:   # oversample
                sampler = RandomOverSampler(
                    sampling_strategy={label: target},
                    random_state=p.random_state
                )
            elif size > target: # undersample
                sampler = RandomUnderSampler(
                    sampling_strategy={label: target},
                    random_state=p.random_state
                )
            else:
                chunks.append(X_c)
                continue

            X_bal, _ = sampler.fit_resample(X_c, y_c)
            chunks.append(X_bal)

        return cp.asarray(np.vstack(chunks))

