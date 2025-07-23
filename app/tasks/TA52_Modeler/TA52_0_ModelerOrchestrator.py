import logging
import traceback
from queue import Queue
from threading import Thread, Lock
import threading
from typing import List, Dict, Any, Optional
import pandas as pd
import gc
from copy import deepcopy
import time
import pandas as pd
import cupy as cp
from itertools import chain



import os
os.environ["CUML_LOG_LEVEL"] = "error"
import cuml

from app.tasks.TA52_Modeler.TA52_A_Preprocessor import TA52_A_Preprocessor
from app.tasks.TA52_Modeler.TA52_B_Modeler import TA52_B_Modeler
from app.tasks.TA52_Modeler.TA52_C_Validator import TA52_C_Validator
from app.utils.common.app.utils.dataModels.Jobs.ModelerJob import ModelerJob, ModelerJobInput, ModelerAttrs


from contextlib import contextmanager



from app.utils.common.app.utils.dataModels.FilterModel.FilterModel import FilterModel


from app.utils.common.app.utils.dataModels.Jobs.ModelerJob import ModelerJob


from app.utils.common.app.utils.SQL.models.production.api_SegmentationResults import SegmentationResults_Out
from app.utils.common.app.utils.SQL.models.production.api_ModellingResults import ModellingResults_Out
from app.utils.common.app.utils.SQL.models.production.api_ClusterEmbeddings import ClusterEmbeddings_Out


from queue import Queue
from threading import Thread, Lock
import threading
import pandas as pd

### utils
from app.tasks.TA52_Modeler.utils.StackDataLoader import StackDataLoader
from app.tasks.TA52_Modeler.utils.StackDataLoaderDB import StackDataLoaderDB
from app.tasks.TA52_Modeler.utils.input_queue import input_queue, output_queue


PERMANENT_CACHE = os.getenv("PERMANENT_CACHE", True)


class TA52_0_ModelerOrchestrator:
    def __init__(self):
        self.stats_list = []
        self.stats_lock = Lock()

    def run(self):
        while True:
            job = input_queue.get()
            try:
                self._execute_job(job)  # Processes fully: preprocess ➔ split ➔ bootstrap ➔ modeler ➔ validator ➔ DB update.
            finally:
                input_queue.task_done()



    def _execute_job(self, job):
        preprocessor = TA52_A_Preprocessor()
        modeler = TA52_B_Modeler()
        validation = TA52_C_Validator()
        finished_jobs = []


        try:
            logging.debug3(f"[PREPROCESS] Job {job.job_uuid} preprocessed with status {job.status}")
            job = preprocessor.run(job)
            logging.debug3(f"[PREPROCESS] Job {job.job_uuid} preprocessed with status {job.status}")
            if job.status == "FAILED":
                logging.debug3(f"[PREPROCESS] Job {job.job_uuid} preprocessed with status {job.status}")

                job = self.clear_job_attrs_except_stats(job)
                job.update_db(fields_to_update=["status", "payload"])
                self._cleanup_job_memory(job)
                return


            # --------------- Create n test and train slices and associated subjobs ---------------

            logging.debug2(f"[SPLIT] Starting splitter for job {job.job_uuid}")
            from app.tasks.TA52_Modeler.utils.create_train_test_subjobs import create_train_test_subjobs

            try:
                    

                subjobs = create_train_test_subjobs(
                    job,
                    n_splits=5,
                    test_size=0.2,
                    random_state=42
                )

                if not isinstance(subjobs, list):
                    if subjobs.status == "FAILED":
                        job = self.clear_job_attrs_except_stats(job)
                        job.update_db(fields_to_update=["status", "payload"])
                        self._cleanup_job_memory(job)
                        return

                logging.debug3(f"[SPLIT] Created {len(subjobs)} subjobs for job {job.job_uuid}")

            except Exception as e:
                logging.debug3(f"[SPLIT] Failed for job {job.job_uuid} with error: {e}")
                job.status = "FAILED"
                job.input.fail_trail.mark("preprocessing", "splitter", f"{type(e).__name__}: {str(e)}")
                job = self.clear_job_attrs_except_stats(job)
                job.update_db(fields_to_update=["status", "payload"])
                self._cleanup_job_memory(job)
                return


            # --------------- Iterate over train/test split subjobs iterations ---------------
            # Determine bootstrap iterations from config:
            bs_config = job.input.preProcessing_instructions.bootstrapping
            n_iter = getattr(bs_config, "n_iterations", 1)

            if n_iter <= 1:
                # No bootstrapping → single pass
                iter_list = [0]
            else:
                iter_list = list(range(1, n_iter + 1))

            results_summary = {}  # Top-level dict to hold all outcomes           
            ran_bjobs = []
            
            for fold_no, subjob in enumerate(subjobs, start=1):
                
                fold_key = f"split_{fold_no}"
                results_summary[fold_key] = {}
                logging.debug2(f"[SPLIT] Processing fold {fold_no}/{len(subjobs)} for job {job.job_uuid}")


            # --------------- Bootstrap iterations ---------------

                for bs_iter in iter_list:
                    iter_key = f"bootstrap_{bs_iter}"
                    logging.debug2(f"[BOOTSTRAP] Iteration {bs_iter}/{len(iter_list)} for fold {fold_no} of job {job.job_uuid}")

                    try:
                        
                        bjob = deepcopy(subjob)  # Sub-subjob = bootstrap iteration on this split
                        bjob.input.bootstrap_iteration = bs_iter

                        bjob.attrs.data_train = self.bootstrap_sample_data_train(
                            data_train=bjob.attrs.data_train,
                            row_count=bs_config.n_samples,
                            random_state=bs_iter  # Use iteration number as seed for reproducibility
                        )



                        if bjob.status == "FAILED":
                            reason = bjob.input.fail_trail.get_last_fail_reason()
                            results_summary[fold_key][iter_key] = f"FAILED: {reason}"
                        else:
                            results_summary[fold_key][iter_key] = "SUCCESS"

                    except Exception as e:
                        results_summary[fold_key][iter_key] = "FAILED: on bootstrap"
                    


                    bjob = modeler.run(bjob)
                    logging.debug3(f"[MODEL] Job {bjob.job_uuid} modelling completed with status {bjob.status}")
                    if bjob.status == "FAILED":
                        logging.debug3(f"[MODEL] Job {bjob.job_uuid} modelling FAILED with reason {bjob.input.fail_trail.get_last_fail_reason()}")

                        reason = bjob.input.fail_trail.get_last_fail_reason()
                        results_summary[fold_key][iter_key] = f"FAILED: {reason}"
                        bjob = self.clear_job_attrs_except_stats(bjob)
                        ran_bjobs.append(bjob)
                        continue                            

                        
                    else:
                        results_summary[fold_key][iter_key] = "SUCCESS"

                    logging.debug2(f"Modelling successful for foldNo {fold_no}/{len(subjobs)} and iteration {bs_iter}/{len(iter_list)} of job {job.job_uuid}")

                    bjob = validation.run(bjob)
                    if bjob.status == "FAILED":
                        results_summary[fold_key][iter_key] = f"FAILED: {bjob.input.fail_trail.get_last_fail_reason}"
                        bjob = self.clear_job_attrs_except_stats(bjob)
                        ran_bjobs.append(bjob)      
                        continue                      
                        
                    else:
                        results_summary[fold_key][iter_key] = "SUCCESS"

                    logging.debug2(f"Validation successful for foldNo {fold_no}/{len(subjobs)} and iteration {bs_iter}/{len(iter_list)} of job {job.job_uuid}")

                    self._harvest_bjob_outputs(job, bjob)  # Merge metrics from bjob into job
                    
                    
                    # ---- FREE MEMORY ON GPU AND HOST ---
                    bjob = self.clear_job_attrs_except_stats(bjob)

                    cp.get_default_memory_pool().free_all_blocks()
                    gc.collect()

                    del bjob


                    

            def store_validation_results_df(job: ModelerJob):
                """
                Store the validation results DataFrame to the database.
                """
                import hashlib
                def _build_validation_hash(row: dict) -> str:
                    """
                    Deterministic hex digest of the identifying fields.
                    """
                    base_string = f"{row['DoE_UUID']}|{row['scope']}|{row['frac']}|{row['label']}|{row['fold_no']}|{row['bootstrap_no']}"
                    return "val_" + hashlib.sha256(base_string.encode("utf-8")).hexdigest()[:10]

                try:
                    if job.attrs.validation_results_df is None or job.attrs.validation_results_df.empty:
                        logging.debug2(f"[VALIDATION] No validation results to store for job {job.job_uuid}")
                        job.status = "FAILED"
                        job.input.fail_trail.mark("validation", "store_results", "No validation results to store")
                        return job
                    df = job.attrs.validation_results_df.copy()
                    #job.attrs.validation_results_df = None  # Clear after storing
                    job.status = "DONE"
                
                    df["validation_UUID"] = df.apply(_build_validation_hash, axis=1)
                    with suppress_logging(logging.ERROR):
                        ModellingResults_Out.store_dataframe(df=df)
                except Exception as e:
                    logging.error(f"[VALIDATION] Failed to store validation results for job {job.job_uuid}: {e}")
                    traceback.print_exc()
                    return job

                return job

            job = store_validation_results_df(job)
            job = self.clear_job_attrs_except_stats(job)
            
            finished_jobs.append(job)
            #self._print_aggregated_summary_stats(finished_jobs)

            summary_df = build_minimal_summary(finished_jobs)
            logging.debug2(summary_df.round(2))

            self._cleanup_job_memory(job)
            job.status = "DONE"
            job.update_db(fields_to_update=["status", "payload",])

        except Exception as e:
            logging.exception(f"[{threading.current_thread().name}]")

    def bootstrap_sample_data_train(self, data_train: cp.ndarray, row_count: int, random_state: int = None) -> cp.ndarray:
        """
        Perform bootstrap sampling on a CuPy `data_train` matrix:
        - Sample `row_count` rows with replacement.
        - Operates fully on GPU.

        Parameters
        ----------
        data_train : cp.ndarray
            CuPy array representing training dataset.

        row_count : int
            Desired number of rows in bootstrap sample.

        random_state : int, optional
            Seed for reproducibility (optional for deterministic runs).

        Returns
        -------
        cp.ndarray
            Bootstrapped dataset (CuPy array, same columns as input).
        """
        if not isinstance(data_train, cp.ndarray):
            raise TypeError("bootstrap_sample_data_train requires a CuPy array as input.")

        num_rows = data_train.shape[0]

        if num_rows == 0:
            raise ValueError("bootstrap_sample_data_train: input data_train has 0 rows.")

        if row_count == 0:
            logging.warning("bootstrap_sample_data_train called with row_count=0 → returning empty array.")
            return data_train[:0]

        rng = cp.random.default_rng(seed=random_state)
        indices = rng.integers(0, num_rows, size=row_count, endpoint=False)
        bootstrapped = data_train[indices]

        # 🔔 Diagnostic logging:
        change = row_count - num_rows
        factor = row_count / num_rows if num_rows > 0 else 0
        logging.debug2(
            f"[BOOTSTRAP] Sampling completed: original rows={num_rows}, sampled rows={row_count} "
            f"(change={change:+d}, factor={factor:.2f}x)"
        )

        return bootstrapped

    def clear_job_attrs_except_stats(self, job):
        """
        Utility to clean a ModelerJob after failure for safe archiving.

        Sets all .attrs fields to None except for .stats, so the job can
        be stored and post-analyzed without holding large or invalid data.

        Parameters
        ----------
        job : ModelerJob
            The job object to clean.
        """

        if not hasattr(job, "attrs"):
            return job

        preserved_stats = job.stats

        # Null all known attrs fields except stats
        job.attrs = type(job.attrs)()  # Re-initialize as empty structure
        job.stats = preserved_stats

        return job

    def _cleanup_job_memory(self, job):
        # Remove heavy attributes
        attrs_to_clear = [
            "raw_data",
            "preProcessed_data",
            "data_num",
            "engineered_data",
            "multi_pca_results",
            "results_cupy",
        ]


        for attr in attrs_to_clear:
            if hasattr(job.attrs, attr):
                try:
                    setattr(job.attrs, attr, None)
                except Exception:
                    pass

        # Clear other general-purpose job data if needed
        #job.input = None
        #job.context = None

        # GPU memory cleanup (CuPy only)
        try:
            import cupy as cp
            cp._default_memory_pool.free_all_blocks()
        except Exception:
            pass  # CuPy not in use or already cleaned

        # Python memory cleanup
        gc.collect()


    # -----------------------------------------------------
    #   Pull results out of a bjob and discard the heavy parts
    # -----------------------------------------------------
    def _harvest_bjob_outputs(self, parent: ModelerJob, child: ModelerJob) -> None:
        """Merge metrics from *child* (one bootstrap) into the
        parent split-job and free GPU / host RAM held by *child*."""

        i = child.input.bootstrap_iteration      # bootstrap #
        f = getattr(child.input, "outer_fold", 0)  # split #

        # ---- 1.  Fail-trail --------------------------------------------------
        parent.input.fail_trail.modelling[f"fold_{f}_bootstrap_{i}"] = \
            dict(child.input.fail_trail.modelling)
        parent.input.fail_trail.validation[f"fold_{f}_bootstrap_{i}"] = \
            dict(child.input.fail_trail.validation)

        # ---- 2.  Stats -------------------------------------------------------
        for stage in ("modelling", "validation"):
            parent.stats.setdefault(stage, {})
            parent.stats[stage][f"fold_{f}_bootstrap_{i}"] = child.stats.get(stage, {})

        # ---- 3.  Validation results DF --------------------------------------
        vdf = getattr(child.attrs, "validation_results_df", None)
        if vdf is not None and not vdf.empty and child.status != "FAILED":
            if parent.attrs.validation_results_df is None:
                parent.attrs.validation_results_df = vdf
            else:
                parent.attrs.validation_results_df = pd.concat(
                    [parent.attrs.validation_results_df, vdf], ignore_index=True
                )



@contextmanager
def suppress_logging(level=logging.ERROR):
    """
    Temporarily suppress logging messages below the specified level.
    """
    logger = logging.getLogger()
    handlers = logger.handlers[:]
    original_levels = [h.level for h in handlers]

    try:
        for h in handlers:
            h.setLevel(level)
        yield
    finally:
        for h, original_level in zip(handlers, original_levels):
            h.setLevel(original_level)

# Utility



def _minimal_record(job):
    stats = job.stats

    # Bootstrap runs
    boots = stats.get("bootstrapping", {}).get("runs", []) or [0]
    pre = stats.get("preprocessing", {})
    modelling = stats.get("modelling", {})
    validation = stats.get("validation", {})

    recs = []
    for i in boots:
        bkey = f"bootstrap_{i}"
        mod = modelling.get(bkey, {})
        val = validation.get(bkey, {}).get("step_A_check_and_prepare", {})

        # Shapes
        shape_start = pre.get("shape_initial", [None, None])
        shape_final = pre.get("step_F_sampling", {}).get("shape_after", shape_start)

        recs.append({
            "resampling_method": pre.get("method"),
            "n_bootstraps": len(boots),
            "bootstrap_samples": shape_final[0],

            "rows_initial": shape_start[0],
            "cols_initial": shape_start[1],
            "rows_final": shape_final[0],
            "cols_final": shape_final[1],
            "rows_dropped": shape_start[0] - shape_final[0] if None not in shape_start + shape_final else None,
            "cols_dropped": shape_start[1] - shape_final[1] if None not in shape_start + shape_final else None,

            "prep_total_s": pre.get("total_elapsed_time"),
            "mod_cfg_s": mod.get("config_check_s"),
            "mod_bin_s": mod.get("binning_s"),
            "mod_dimred_s": mod.get("dim_reduction_s"),
            "val_total_s": val.get("elapsed_time"),
        })

    return recs


def build_minimal_summary(jobs: list) -> pd.DataFrame:
    rows = list(chain.from_iterable(_minimal_record(j) for j in jobs))
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Group + aggregate
    group_keys = ["resampling_method", "n_bootstraps", "bootstrap_samples"]
    duration_cols = ["prep_total_s", "mod_cfg_s", "mod_bin_s", "mod_dimred_s", "val_total_s"]
    shape_cols = ["rows_initial", "cols_initial", "rows_final", "cols_final", "rows_dropped", "cols_dropped"]

    summary = (
        df
        .groupby(group_keys)
        .agg({col: "mean" for col in duration_cols + shape_cols})
        .reset_index()
    )

    return summary
