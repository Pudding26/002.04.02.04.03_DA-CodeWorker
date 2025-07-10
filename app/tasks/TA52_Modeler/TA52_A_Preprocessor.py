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



## Variables
### QM_Sampling
MIN_CLASSES = 2
MIN_PER_CLASS = 5
MIN_ROWS = 40
## QM_Scaling
SCALING_VAIRANCE_TRESHOLD = 1e-8
MIN_ROWS_SCALING = 2
NAN_COL_FRACTION_TRESHOLD = 0.5  # Fraction of NaNs in a column above which it is dropped
## QM_Binning
MUTATION_DROP_TRESHOLD = 0.75  # Trehsold of how many rows where dropped from the inital dataset after all qm and preprocessing steps to allow high correlation between inital DoE Recipe and actual modeled data

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
        # ------------------------------ Step1: Check integrity of instructions
        @staticmethod
        def step_A_prepare_and_check_instructions(job):
            """
            Step A – Prepare & Check Instructions

            Validates the structure of the preprocessing config and unpacks
            method/submethod/subsubmethod. Fails early if config is malformed.
            Writes preliminary preprocessing stats and marks fail_trail.

            Returns
            -------
            job : ModelerJob
                - Marked FAILED if instruction block is malformed.
                - Otherwise, stats["preprocessing"] will contain early metadata.
            """
            cfg = job.input.preProcessing_instructions
            raw_data = job.attrs.raw_data
            shape_initial = raw_data.shape

            try:
                # ⬇️ Updated to flattened format: resampling.method + its config
                res_cfg = cfg.resampling
                if res_cfg is None:
                    raise AttributeError("No 'resampling' section found in preprocessing config")

                if res_cfg.method is None:
                    raise AttributeError("No 'method' specified under 'resampling'")

                sampler_method = res_cfg.method
                sampler_cfg = getattr(res_cfg, sampler_method, None)
                if sampler_cfg is None:
                    raise AttributeError(f"Missing sampler config block for method '{sampler_method}'")

                # ⬇️ Stats (initial)
                job.stats.setdefault("preprocessing", {})
                job.stats["preprocessing"].update({
                    "method": sampler_method,
                    "shape_initial": shape_initial,
                    "elapsed_time": 0.0           # placeholder
                })

                job.context = f"Preprocessing with method: {sampler_method}"
                logging.debug2(job.context)
                job.input.fail_trail.mark("preprocessing", "check_inst_integrity", "passed")
                return job

            except AttributeError as e:
                
                job.attrs.preProcessed_data = raw_data
                job.status = "FAILED"
                job.input.fail_trail.mark("preprocessing", "check_inst_integrity", f"incomplete: {str(e)}")
                job.stats["preprocessing"] = {
                    "method": getattr(cfg, "method", None),
                    "submethod": None,
                    "subsubmethod": None,
                    "shape_initial": shape_initial,
                    "elapsed_time": 0.0
                }
                
                return job


        job = step_A_prepare_and_check_instructions(job)
        if job.status == "FAILED":
            return job

        # ------------------------------ Step2: Prepare numeric GPU data
        def step_B_prepare_GPU_data(job):
            """
            Step B – Prepare GPU Data

            Converts the raw input data (`job.attrs.raw_data`) into a numeric CuPy array 
            stored in `job.attrs.data_num`. Also introduces a column encoder dictionary 
            (`job.attrs.encoder.cols`) which maps feature names to column indices. This 
            encoder is critical for all downstream feature selection, masking, and 
            weighting steps.

            This step handles:
            ▸ Optional debug expansion of raw data
            ▸ Transformation of raw data to GPU-backed format (CuPy)
            ▸ Encoder dictionary generation for feature indexing
            ▸ Logging and error handling
            ▸ FailTrail marking and job.stats recording

            Returns
            -------
            job : ModelerJob
                - With `attrs.data_num` and `attrs.encoder.cols` set if successful
                - Marked as FAILED if conversion fails

            Notes
            -----
            - The encoder mapping (`job.attrs.encoder.cols`) must be preserved 
            for all column-aware operations such as blacklisting, weighting, scaling, etc.
            - Use `job.attrs.colname_encoder` for ordered list of column names (optional)
            """

            # Uncomment for synthetic testing:
            # TA52_A_Preprocessor.expand_raw_data_debug(job, factor=500)
            def update_step_stats(job):

                shape_after = list(job.attrs.data_num.shape)
                elapsed = round(time.time() - t0, 3)

                step_key = "step_B_prepareGPU_data"
                step_stats = {
                    "elapsed_time": elapsed,
                    "shape_before": shape_before,
                    "shape_after": shape_after,
                    "shape_change": [shape_after[0] - shape_before[0], shape_after[1] - shape_before[1]],
                }

                # Safely initialize the block
                job.stats.setdefault("preprocessing", {})
                job.stats["preprocessing"][step_key] = step_stats

                # Update total time (cumulative)
                previous_total = job.stats["preprocessing"].get("total_elapsed_time", 0.0)
                job.stats["preprocessing"]["total_elapsed_time"] = round(previous_total + elapsed, 3)

                return job
            
            t0 = time.time()
            
            
            try:
                logging.debug2("Preparing numeric GPU data...")
                shape_before = list(job.attrs.raw_data.shape)
                job = TA52_A_Preprocessor._prepare_numeric_gpu_data(job)
                job = update_step_stats(job)

            except Exception as e:
                logging.error(
                    f"[PREPROCESS ERROR] Failed to prepare numeric GPU data for job {getattr(job, 'id', 'unknown')}",
                    exc_info=True
                )
                job.status = "FAILED"
                job.input.fail_trail.mark("preprocessing", "prepare_numeric_gpu_data", f"failed: {str(e)}")
                return job

            logging.debug2("Numeric GPU data prepared.")

            
            # Record stats after preparation

            

            job.input.fail_trail.mark("preprocessing", "create_GPU_data", "passed")

            return job


        job = step_B_prepare_GPU_data(job)
        if job.status == "FAILED":
            return job


        # ------------------------------ Step01: Scaling
        scaling_start = time.time()
        logging.debug1("Scaling data...")
    
        def step_01_scaling(job: ModelerJob) -> ModelerJob:
            """
            Step 01 – Scaling: Prepares numeric data and applies MinMax normalization.

            This wrapper performs the full preprocessing step for feature scaling:
            ▸ Cleans the dataset using `qm_pre_scaling`:
                - Drops non-blacklisted columns with zero variance or infinite values
                - Leaves NaNs intact (assumed to be handled by later weighting logic)
            ▸ Applies `minmax_scaler` to all remaining features not blacklisted,
                scaling them to a fixed range (default: [0, 1])

            If any critical issue is detected during cleanup (e.g., too few rows remain),
            the job is marked as FAILED and returned without scaling.

            Parameters
            ----------
            job : ModelerJob
                A job object containing:
                - `attrs.data_num` : CuPy ndarray of encoded numeric data
                - `attrs.encoder.cols`, `attrs.blacklist` : Metadata required for filtering
                - `fail_trail` : Pydantic model to trace step outcomes

            Returns
            -------
            job : ModelerJob
                The modified job object. Scaled data is stored in `attrs.data_num`.
                If the job failed quality checks, its status will be `"FAILED"` and
                `attrs.data_num` will remain unscaled.

            Notes
            -----
            - This function assumes `fail_trail` is already initialized on the job.
            - This is a standardized scaling step used in the TA52 pipeline.
            - Blacklisted columns (e.g., ID or categorical scope variables) are preserved
            and never scaled.

            Dependencies
            ------------
            - qm_pre_scaling()
            - minmax_scaler()
            """


            from app.tasks.TA52_Modeler.utils.TA52_A_utils.qm_pre_scaling import qm_pre_scaling
            from app.tasks.TA52_Modeler.utils.TA52_A_utils.minmax_scaler import minmax_scaler
            from app.tasks.TA52_Modeler.utils.TA52_A_utils.drop_high_nan_cols import drop_high_nan_cols



            t0 = time.time()
            shape_before = list(job.attrs.data_num.shape)

            # --- Quality check: drop problematic cols before scaling
            job = qm_pre_scaling(job, variance_threshold=SCALING_VAIRANCE_TRESHOLD, min_rows=MIN_ROWS_SCALING)
            if job.status == "FAILED":
                return job

            # --- Apply actual scaling and drop nan cols
            try:
                job = drop_high_nan_cols(
                    job,
                    threshold=NAN_COL_FRACTION_TRESHOLD
                )
                job.attrs.data_num = minmax_scaler(job, feature_range=(0, 1))
            except Exception as e:
                logging.error(
                    f"[PREPROCESS ERROR] Failed to apply MinMax scaling for job {getattr(job, 'id', 'unknown')}",
                    exc_info=True
                )
                job.status = "FAILED"
                job.input.fail_trail.mark("preprocessing", "step_01_scaling", f"failed: {str(e)}")
                return job

            # --- Logging stats
            shape_after = list(job.attrs.data_num.shape)
            elapsed = round(time.time() - t0, 3)

            job.stats["preprocessing"]["step_01_scaling"] = {
                "elapsed_time": elapsed,
                "shape_initial": shape_before,
                "shape_after": shape_after,
                "shape_change": [shape_after[0] - shape_before[0], shape_after[1] - shape_before[1]],
            }

            # --- Update total time
            job.stats["preprocessing"]["total_elapsed_time"] = round(
                job.stats["preprocessing"].get("total_elapsed_time", 0.0) + elapsed, 3
            )

            return job

            
        job = step_01_scaling(job)
        if job.status == "FAILED":
            return job
        scaling_end = time.time()
        # ------------------------------ Step02: Handling NaNs and bin weighting
        weighting_start = time.time()
        logging.debug2("Handling NaNs and applying bin weighting...")

        
        def step02_bin_weigting(job):
            """
            Step 02 – Bin Fraction Weighting

            Applies per-bin fractional weighting to features based on encoded column names.
            Each feature group is scaled by its corresponding "_fraction" column using the 
            logic defined in `bin_fraction_weighting()`.

            If an error occurs during weighting (e.g., missing fractions or malformed column names),
            the job is marked as FAILED and returned unmodified.

            Parameters
            ----------
            job : ModelerJob
                The current job, with scaled data and column encodings.

            Returns
            -------
            job : ModelerJob
                Modified job with weighted features, or marked as FAILED on error.

            See Also
            --------
            - bin_fraction_weighting : Implements the core weighting logic
            - step01_scaling : Should precede this step
            """


            from app.tasks.TA52_Modeler.utils.TA52_A_utils.bin_fraction_weighting import bin_fraction_weighting

            t0 = time.time()
            shape_before = list(job.attrs.data_num.shape)

            try:
                # --- Apply bin weighting
                job.attrs.data_num = bin_fraction_weighting(job)

                # --- Drop resulting NaNs
                job, dropped = TA52_A_Preprocessor._drop_nans(job)
                job.stats["preprocessing"]["dropped_nans_after_binning"] = dropped

                # --- Final check: ensure not too many rows dropped
                job = TA52_A_Preprocessor._check_dropped_fraction(
                    job,
                    shape_before,
                    dropped=dropped,
                    threshold=MUTATION_DROP_TRESHOLD
                )
                if job.status == "FAILED":
                    job.input.fail_trail.mark("preprocessing", "bin_fraction_weighting", "failed: excessive NaN rows dropped")
                    return job

            except Exception as e:
                logging.error(
                    f"[PREPROCESS ERROR] Failed to apply bin fraction weighting for job {getattr(job, 'id', 'unknown')}",
                    exc_info=True
                )
                job.status = "FAILED"
                job.input.fail_trail.mark("preprocessing", "bin_fraction_weighting", f"failed: {str(e)}")
                return job

            # --- Mark step success
            shape_after = list(job.attrs.data_num.shape)
            elapsed = round(time.time() - t0, 3)

            job.stats["preprocessing"]["step_E_bin_weighting"] = {
                "elapsed_time": elapsed,
                "shape_initial": shape_before,
                "shape_after": shape_after,
                "shape_change": [shape_after[0] - shape_before[0], shape_after[1] - shape_before[1]],
                "dropped_nans": dropped
            }

            job.stats["preprocessing"]["total_elapsed_time"] = round(
                job.stats["preprocessing"].get("total_elapsed_time", 0.0) + elapsed, 3
            )

            job.input.fail_trail.mark("preprocessing", "bin_fraction_weighting", "passed")
            return job

        job = step02_bin_weigting(job)
        weighting_end = time.time()
        if job.status == "FAILED":
            return job
        
        logging.debug2("Data scaling and NaN handling completed.")


        # ------------------------------ Step03: Sampling

        @staticmethod
        def step03_sampling(job):
            """
            Step 03 – Sampling: Applies the specified sampling method to the dataset.

            This step performs data sampling based on the preProcessing_instructions:
            ▸ Uses the configured method (e.g., resampling, undersampling, oversampling)
            ▸ Applies the submethod and subsubmethod as defined in the job's input config
            ▸ Handles quality management before sampling to ensure data integrity

            Parameters
            ----------
            job : ModelerJob
                The current job with:
                - `attrs.data_num`: CuPy ndarray of numeric data
                - `input.preProcessing_instructions`: Configuration for sampling

            Returns
            -------
            job : ModelerJob
                Modified job with sampled data in `attrs.data_num`, or marked as FAILED on error.
            """

            from app.tasks.TA52_Modeler.utils.TA52_A_utils.qm_pre_sampling import qm_pre_sampling

            from app.tasks.TA52_Modeler.utils.TA52_A_utils.sampler.random_sampler import random_sampler
            from app.tasks.TA52_Modeler.utils.TA52_A_utils.sampler.SMOTE_sampler import smote_sampler

            t0 = time.time()
            shape_before = list(job.attrs.data_num.shape)



            try:
                logging.debug2("Starting sampling...")
                logging.debug1("Pre-sampling quality management...")

                # --- Quality check before sampling
                job = qm_pre_sampling(
                    job,
                    min_classes=MIN_CLASSES,
                    min_per_class=MIN_PER_CLASS,
                    min_rows=MIN_ROWS
                )
                if job.status == "FAILED":
                    return job

                job.input.fail_trail.mark("preprocessing", "qm_pre_sampling", "passed")
                logging.debug1("Pre-sampling quality management passed.")

                # --- Retrieve sampling config from flat structure
                cfg = job.input.preProcessing_instructions.resampling
                if not cfg or not cfg.method:
                    # ⛔ No sampling configured – skip this step
                    logging.debug2("No sampling method configured. Skipping sampling step.")
                    elapsed = round(time.time() - t0, 3)
                    job.stats["preprocessing"]["step_F_sampling"] = {
                        "elapsed_time": elapsed,
                        "shape_initial": shape_before,
                        "shape_after": shape_before,
                        "shape_change": [0, 0],
                        "sampler_method": "none",
                        "scope": job.input.scope
                    }
                    job.stats["preprocessing"]["total_elapsed_time"] = round(
                        job.stats["preprocessing"].get("total_elapsed_time", 0.0) + elapsed, 3
                    )
                    job.input.fail_trail.mark("preprocessing", "sampling", "skipped")
                    return job

                sampler_method = cfg.method
                sampler_cfg = getattr(cfg, sampler_method, None)
                if sampler_cfg is None:
                    raise ValueError(f"Sampler config block for '{sampler_method}' is missing")

                logging.debug2(f"Applying sampler: {sampler_method} with scope: {job.input.scope}")

                # --- Dispatch to actual sampling implementation
                match sampler_method:
                    case "RandomSampler":
                        X_out = random_sampler(job)
                    case "SMOTESampler":
                        X_out = smote_sampler(job)
                    case _:
                        raise ValueError(f"Unsupported sampling method: {sampler_method}")

                # --- Apply result
                job.attrs.data_num = X_out
                shape_after = list(job.attrs.data_num.shape)
                elapsed = round(time.time() - t0, 3)

                # --- Log stats
                job.stats["preprocessing"]["step_F_sampling"] = {
                    "elapsed_time": elapsed,
                    "shape_initial": shape_before,
                    "shape_after": shape_after,
                    "shape_change": [shape_after[0] - shape_before[0], shape_after[1] - shape_before[1]],
                    "sampler_method": sampler_method,
                    "scope": job.input.scope
                }

                job.stats["preprocessing"]["total_elapsed_time"] = round(
                    job.stats["preprocessing"].get("total_elapsed_time", 0.0) + elapsed, 3
                )

                job.input.fail_trail.mark("preprocessing", "sampling", "passed")
                return job


            except Exception as e:
                logging.error(
                    f"[PREPROCESS ERROR] Sampling step failed for job {getattr(job, 'id', 'unknown')}",
                    exc_info=True
                )
                job.status = "FAILED"
                job.input.fail_trail.mark("preprocessing", "sampling", f"failed: {str(e)}")
                return job

            

        job = step03_sampling(job)
        if job.status == "FAILED":
            return job

        # ---------------------------------------------------------------- stats
        @staticmethod
        def step_C_wrapup(job):
            """
            Step C – Finalize preprocessing

            Performs final checks and assignments before returning the job:
            - Drops remaining NaNs (if needed)
            - Verifies post-sampling row loss
            - Stores `preProcessed_data` for downstream stages
            - Updates context and summary logs

            Parameters
            ----------
            job : ModelerJob
                The job after all preprocessing steps
            shape_before : list
                Shape before sampling step (used to detect excessive row loss)

            Returns
            -------
            job : ModelerJob
                Finalized job object or marked as FAILED
            """

            shape_before = list(job.attrs.data_num.shape)

            # Drop any NaNs left
            job, dropped = TA52_A_Preprocessor._drop_nans(job)
            job = TA52_A_Preprocessor._check_dropped_fraction(job, shape_before, dropped)
            job.stats["preprocessing"]["dropped_nans_after_sampling"] = dropped

            if job.status == "FAILED":
                logging.warning(
                    f"[PREPROCESS] Job {job.job_uuid} failed due to excessive NaN rows dropped ({dropped}) after sampling."
                )
                return job

            # Final assignment
            job.attrs.preProcessed_data = job.attrs.data_num

            # Set job context based on method (if configured)
            method = "none"
            try:
                res_cfg = job.input.preProcessing_instructions.resampling
                if res_cfg and res_cfg.method:
                    method = res_cfg.method
            except Exception:
                pass

            job.context = f"SUCCESS of: Preprocessing with method: {method}"

            logging.debug1("SUCCESS")
            logging.debug3(
                f"Preprocessing completed in {job.stats['preprocessing'].get('total_elapsed_time', 'N/A')} seconds"
            )

            return job


        
        job = step_C_wrapup(job)
        return job





#----------
# UTILITY
#----------

    @staticmethod
    def _drop_nans(job):
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
    def _prepare_numeric_gpu_data(job) -> None:
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

        #index_keep = INDEX_COLS[:scope_idx + 1]        # ['session', 'campaign']
        #index_drop = INDEX_COLS[scope_idx + 1:]        # ['shotID']
        #df_work    = df_raw.drop(columns=index_drop)   # drop finer levels

        df_work = df_raw
        
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

        return job

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


    # ─────────────────────────────────────────────────────────── scalers ────
    @staticmethod
    def _standard_scaler(job):
        from cuml.preprocessing import StandardScaler
        cfg = job.input.preProcessing_instructions
        p   = cfg.scaling.standardization.StandardScaler
        scaler = StandardScaler(with_mean=p.with_mean, with_std=p.with_std)
        return scaler.fit_transform(job.attrs.data_num)

    # DEBUGGING
    @staticmethod
    def expand_raw_data_debug(job, factor: int) -> None:
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
