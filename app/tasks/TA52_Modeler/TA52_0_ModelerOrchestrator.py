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

class TA52_0_ModelerOrchestrator:
    def __init__(self):
        self.input_queue = None
        self.output_queue = None
        self.num_workers = 4
        self.results_df = pd.DataFrame()
        self.stats_list = []
        self.stats_lock = Lock()

    def run(self, job_df_raw: pd.DataFrame, num_workers: int = 1):
        self.num_workers = num_workers
        self.input_queue = Queue(maxsize=2 * num_workers)
        self.output_queue = Queue(maxsize=2 * num_workers)
        self.raw_input_df = job_df_raw
        self.results_df = pd.DataFrame()
        self.stats_list = []

        self._start_pipeline(job_df_raw)

        return self.results_df

    def _start_pipeline(self, job_df_raw):
        # Start Loader in a thread
        loader_thread = Thread(target=self._load_jobs, name="LoaderThread", daemon=True)
        loader_thread.start()

        # Start Storer in a thread
        storer_thread = Thread(target=self._store_results, name="SaverThread", daemon=True)
        storer_thread.start()

        # Execute jobs in main thread
        self._execute_jobs()

        # Wait for loader and storer to finish
        loader_thread.join()
        self.output_queue.join()
        storer_thread.join()

    def _load_jobs(self):
        job_list = self.convert_raw_dataframe(self.raw_input_df)

        data_loader = StackDataLoader(api_model_cls=SegmentationResults_Out)

        for i, job in enumerate(job_list):
            try:
                job.attrs.raw_data = data_loader.load_for_job(job.input.stackIDs)

                if i % 10 == 0 or i == len(job_list) - 1:
                    logging.debug2(f"[LOADER] Enriched job {i+1}/{len(job_list)} with raw_data")

                self.input_queue.put(job)

            except Exception as e:
                logging.warning(f"[LOADER WARNING] Failed to attach raw_data to Job {job.job_uuid}: {str(e)}")



        # ✅ Send one poison pill per worker
        for _ in range(self.num_workers):
            self.input_queue.put(None)



    def _execute_jobs(self):
        preprocessor = TA52_A_Preprocessor()
        modeler = TA52_B_Modeler()
        validation = TA52_C_Validator()
        finished_jobs = []


        while True:
            job = self.input_queue.get()
            if job is None:
                self.output_queue.put(None)
                self.input_queue.task_done()
                break

            try:
                job = preprocessor.run(job)
                if job.status == "FAILED":
                    job = self.clear_job_attrs_except_stats(job)
                    job.update_db(fields_to_update=["status", "payload"])
                    self._cleanup_job_memory(job)
                    continue



                # BOOTSTRAPPING
                from app.tasks.TA52_Modeler.utils.TA52_A_utils.sampler.bootstrap_sampler import bootstrap_sampler
                
                bootstrap_cfg = job.input.preProcessing_instructions.bootstrapping

                n_iter = bootstrap_cfg.n_iterations
                n_samples = bootstrap_cfg.n_samples
                successful_runs = []
                failed_runs = []
                subjobs = []

                if n_iter == 1:
                    # No bootstrapping – just one standard run
                    i_list = [0]
                else:
                    # Bootstrapping – start at 1
                    i_list = list(range(1, n_iter + 1))

                for i in i_list:
                    logging.debug2(f"[BOOTSTRAP] Starting iteration {i} for job {job.job_uuid}")

                    subjob = deepcopy(job)
                    subjob.input.bootstrap_iteration = i

                    try:
                        time_0 = time.time()
                        subjob = bootstrap_sampler(subjob)
                        shape_before = subjob.attrs.data_num.shape



                        ellapsed_time = round(time.time() - time_0, 4)
                        shape_after = subjob.attrs.data_num.shape

                        stats = {
                            "bootstrap_iteration": i,
                            "n_samples": n_samples,
                            "shape_before": shape_before,
                            "shape_after": shape_after,
                            "ellapsed_time": ellapsed_time,
                        }
                        subjob.stats["preprocessing"]["bootstrapping"] = stats


                        logging.debug2(f"Bootstrapping succesful. new shape: {subjob.attrs.data_num.shape}")

                    except Exception as e:
                        logging.debug2(f"[BOOTSTRAP] Bootstrapping failed for iteration {i} with error: {e}")
                        subjob.status = "FAILED"
                        subjob.input.fail_trail.mark("preprocessing", "bootstrap_sampler", f"{type(e).__name__}: {str(e)}")
                        failed_runs.append(i)
                        subjobs.append(subjob)
                        continue
                
                    subjob = modeler.run(subjob)
                    
                    if subjob.status == "FAILED":
                        logging.debug2(f"[BOOTSTRAP] Modelling failed for iteration {i} with error: {subjob.input.fail_trail.modelling}")
                        subjob = self.clear_job_attrs_except_stats(subjob)
                        self._cleanup_job_memory(subjob)
                        failed_runs.append(i)
                        subjobs.append(subjob)

                        continue
                    logging.debug2(f"[BOOTSTRAP] Modelling successful for iteration {i}")

                    subjob = validation.run(subjob)
                    if subjob.status == "FAILED":
                        subjob = self.clear_job_attrs_except_stats(subjob)
                        self._cleanup_job_memory(subjob)
                        failed_runs.append(i)
                        continue
                    else:
                        successful_runs.append(i)

                    subjobs.append(subjob)

                job.stats["bootstrapping"] = {
                    "runs": [j.input.bootstrap_iteration for j in subjobs],
                    "outcomes": {
                        "success": successful_runs,
                        "failed": failed_runs
                    },
                    "success_rate": round(len(successful_runs) / n_iter, 3),
                }

                job.stats["modelling"] = {}
                job.stats["validation"] = {}



                for subjob in subjobs:
                    i = subjob.input.bootstrap_iteration
                    
                    job.input.fail_trail.modelling[f"bootstrap_{i}"] = dict(subjob.input.fail_trail.modelling)
                    job.input.fail_trail.validation[f"bootstrap_{i}"] = dict(subjob.input.fail_trail.validation)

                
                    if subjob.status == "FAILED":
                        continue

                    job.stats["modelling"][f"bootstrap_{i}"] = subjob.stats.get("modelling", {})
                    job.stats["validation"][f"bootstrap_{i}"] = subjob.stats.get("validation", {})



                    job.attrs.validation_results_df = pd.concat(
                        [job.attrs.validation_results_df, subjob.attrs.validation_results_df],
                        ignore_index=True,
                    ) if job.attrs.validation_results_df is not None else subjob.attrs.validation_results_df


                def store_validation_results_df(job: ModelerJob):
                    """
                    Store the validation results DataFrame to the database.
                    """
                    import hashlib
                    def _build_validation_hash(row: dict) -> str:
                        """
                        Deterministic hex digest of the identifying fields.
                        """
                        base_string = f"{row['DoE_UUID']}|{row['scope']}|{row['frac']}|{row['label']}|{row['bootstrap']}"
                        return "val_" + hashlib.sha256(base_string.encode("utf-8")).hexdigest()[:10]

                    try:
                        df = job.attrs.validation_results_df.copy()
                    
                        df["validation_UUID"] = df.apply(_build_validation_hash, axis=1)
                        with suppress_logging(logging.ERROR):
                            ModellingResults_Out.store_dataframe(df=df)
                    except Exception as e:
                        logging.error(f"[VALIDATION] Failed to store validation results for job {job.job_uuid}: {e}")
                        traceback.print_exc()
                        return job

                    return job

                job = store_validation_results_df(job)
                    
                







                


                finished_jobs.append(job)
                #self._print_aggregated_summary_stats(finished_jobs)

                summary_df = build_minimal_summary(finished_jobs)
                logging.debug2(summary_df.round(2))

                self._cleanup_job_memory(job)
                #job.update_db(fields_to_update=["status", "payload",])

            except Exception as e:
                logging.exception(f"[{threading.current_thread().name}]")

            finally:
                self.input_queue.task_done()


    def convert_raw_dataframe(self, job_df_raw: pd.DataFrame) -> List[ModelerJob]:
        jobs = []
        total_jobs = len(job_df_raw)
        log_interval = 10

        def find_scope_in_dict(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    if k == "scope" and v is not None:
                        return v
                    if isinstance(v, (dict, list)):
                        found = find_scope_in_dict(v)
                        if found is not None:
                            return found
            elif isinstance(d, list):
                for item in d:
                    found = find_scope_in_dict(item)
                    if found is not None:
                        return found
            return None



        for i, (_, row) in enumerate(job_df_raw.iterrows()):
            try:
                payload: Dict[str, Any] = row["payload"]
                inp = ModelerJobInput(
                    stackIDs=payload["input"]["stackIDs"],
                    preProcessing_instructions=payload["input"].get("preProcessing_instructions", {}),
                    metricModel_instructions=payload["input"].get("metricModel_instructions", {}),
                    preProcessingNo=payload["input"]["preProcessingNo"],
                    metricModelNo=payload["input"]["metricModelNo"],
                    scope=payload["input"]["scope"],
                )
                job = ModelerJob(
                    job_uuid=row["job_uuid"],
                    job_type=row["job_type"],
                    status=row["status"],
                    attempts=row["attempts"],
                    next_retry=row["next_retry"],
                    created=row["created"],
                    updated=row["updated"],
                    parent_job_uuids=row["parent_job_uuids"],
                    input=inp,
                    attrs=ModelerAttrs(
                        preprocessed_data=None,  # Will be set later
                        raw_data=None,  # Will be set later
                        model_results=None  # Will be set later
                    )
                )
                jobs.append(job)

                if i % log_interval == 0 or i == total_jobs - 1:
                    logging.debug2(f"[LOADER] Parsed {i+1}/{total_jobs} jobs")

            except Exception as e:
                logging.warning(f"[LOADER WARNING] Failed to parse row {i}: {str(e)}")

        return jobs

    def _store_results(self):
        
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


        finished_workers = 0
        results = []

        counter = 0

        while True:
            job = self.output_queue.get()
            if job is None:
                finished_workers += 1
                self.output_queue.task_done()
                if finished_workers == self.num_workers:
                    break
                continue

            if job.attrs.result_df is not None:
                with suppress_logging(logging.ERROR):
                    ModellingResults_Out.store_dataframe(df=job.attrs.result_df)
                job.attrs.result_df = None
            if job.attrs.featureClusterMap is not None:
                with suppress_logging(logging.ERROR):
                    ClusterEmbeddings_Out.store_dataframe(df=job.attrs.featureClusterMap)
                job.attrs.featureClusterMap = None

            job.status = "DONE"
            job.update_db(fields_to_update=["status"])
            
            if counter % 10 == 0:
                logging.debug2(f"[STORER] Stored job {job.job_uuid} results to DB")

            self.output_queue.task_done()




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


    def _print_summary_df(self, stats_list, key: str):
        try:
            stats_df = pd.DataFrame(stats_list)
            if key in stats_df.columns:
                print(f"\n[{threading.current_thread().name}] Summary for {key}:")
                print(stats_df[[key]].describe(include='all'))
            else:
                print(f"\n[{threading.current_thread().name}] No data for key: {key}")
        except Exception as e:
            print(f"[{threading.current_thread().name}] Summary error for {key}: {e}")



    def convert_raw_dataframe_debug(self, job_df_raw: pd.DataFrame) -> List[ModelerJob]:
        import yaml
        from app.utils.common.app.utils.dataModels.Jobs.ModelerJob import PreProcessingAttributes, MetricModelAttributes

        # Load config for actual injection
        with open("app/config/temp_modelcofig.yaml", "r") as f:
            config_yaml = yaml.safe_load(f)
            def purify_none(value):
                if isinstance(value, dict):
                    return {k: purify_none(v) for k, v in value.items()}
                elif isinstance(value, list):
                    return [purify_none(v) for v in value]
                elif value in ("None", "none", "null", "Null"):
                    return None
                return value
            config_yaml = purify_none(config_yaml)

        jobs = []
        total_jobs = len(job_df_raw)
        log_interval = 10


        for i, (_, row) in enumerate(job_df_raw.iterrows()):
            try:
                payload: Dict[str, Any] = row["payload"]

                # Inject real instruction sets from config
                prep_key = row["temp_preProcessingNo"]
                mm_key = row["temp_metricModelNo"]
                prep_cfg = config_yaml.get(prep_key, {})
                mm_cfg = config_yaml.get(mm_key, {})

                inp = ModelerJobInput(
                    stackIDs=payload["input"]["stackIDs"],
                    preProcessing_instructions=PreProcessingAttributes.parse_obj(prep_cfg),
                    metricModel_instructions=MetricModelAttributes.parse_obj(mm_cfg),
                    preProcessingNo=prep_key,
                    metricModelNo=mm_key,
                )


                job = ModelerJob(
                    job_uuid=row["job_uuid"],
                    job_type=row["job_type"],
                    status=row["status"],
                    attempts=row["attempts"],
                    next_retry=row["next_retry"],
                    created=row["created"],
                    updated=row["updated"],
                    parent_job_uuids=row["parent_job_uuids"],
                    input=inp,
                    attrs=ModelerAttrs(
                        preprocessed_data=None,
                        raw_data=None,
                        model_results=None,
                    )
                )
                jobs.append(job)


                if i % log_interval == 0 or i == total_jobs - 1:
                    logging.debug2(f"[LOADER] Parsed {i+1}/{total_jobs} jobs")

            except Exception as e:
                logging.warning(f"[LOADER WARNING] Failed to parse row {i}: {str(e)}")

        return jobs


    def _print_aggregated_summary_stats(self, jobs: List[ModelerJob]):
        """
        Aggregates stats from multiple jobs and prints an average summary,
        grouped by (resampling_method, n_bootstraps, bootstrap_samples).
        """
        import pandas as pd
        from collections import defaultdict

        aggregated = defaultdict(list)

        for job in jobs:
            try:
                pre = job.stats.get("preprocessing", {})
                boot = job.stats.get("bootstrapping", {})
                mod = job.stats.get("modelling", {})

                resampling_method = pre.get("subsubmethod", "unknown")
                n_boot = boot.get("runs", [])
                n_iter = len(n_boot)
                n_samples = (
                    job.input.preProcessing_instructions.bootstrapping.n_samples
                    if hasattr(job.input, "preProcessing_instructions")
                    else None
                )

                group_key = (resampling_method, n_iter, n_samples)

                record = {
                    "prep_total_s": pre.get("total_elapsed_time"),
                    "prep_qm_drop_cols": pre.get("qm_pre_scaling", {}).get("num_dropped"),
                    "prep_qm_remain_cols": pre.get("qm_pre_scaling", {}).get("num_remaining"),
                    "prep_qm_drop_rows": pre.get("qm_pre_sampling", {}).get("dropped_rows"),
                    "prep_qm_remain_rows": pre.get("qm_pre_sampling", {}).get("remaining_rows"),
                    "prep_bin_drop_nans": pre.get("step_E_bin_weighting", {}).get("dropped_nans"),
                    "mod_config_s": mod.get("config_check_s"),
                    "mod_binning_s": mod.get("binning_s"),
                    "mod_dimred_s": mod.get("dim_reduction_s"),
                    "bootstrap_success_rate": boot.get("success_rate")
                }

                aggregated[group_key].append(record)

            except Exception as e:
                logging.warning(f"[SUMMARY] Failed to aggregate job {getattr(job, 'job_uuid', 'unknown')}: {e}")

        # Flatten and average
        summary_rows = []
        for (method, n_iter, n_samples), records in aggregated.items():
            df = pd.DataFrame(records)
            means = df.mean(numeric_only=True).to_dict()
            means.update({
                "resampling_method": method,
                "n_bootstraps": n_iter,
                "bootstrap_samples": n_samples,
                "n_jobs": len(records),
            })
            summary_rows.append(means)

        if not summary_rows:
            print("No stats available.")
            return

        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df[
            [
                "resampling_method",
                "n_bootstraps",
                "bootstrap_samples",
                "n_jobs",
                "prep_total_s",
                "prep_qm_drop_cols",
                "prep_qm_remain_cols",
                "prep_qm_drop_rows",
                "prep_qm_remain_rows",
                "prep_bin_drop_nans",
                "mod_config_s",
                "mod_binning_s",
                "mod_dimred_s",
                "bootstrap_success_rate",
            ]
        ]

        pd.set_option("display.max_columns", None)
        print("\n=== Aggregated Pipeline Summary ===\n")
        print(summary_df.round(3))



    # ------------------------------------------------------------------
    #  DEBUG helper ─ multiply raw dataframe by PP×MM combinations
    # ------------------------------------------------------------------
    def _debug_jobcreation(self, raw_input_df: pd.DataFrame, config_path: str) -> pd.DataFrame:
        import uuid
        import yaml
        import pandas as pd
        from copy import deepcopy
        import ast

        def _parse_payload(row_payload):
            if isinstance(row_payload, str):
                try:
                    return ast.literal_eval(row_payload)
                except Exception as e:
                    print(f"[ERROR] Failed to parse payload: {e}")
                    return {}
            return row_payload

        with open(config_path, "r") as f:
            config_yaml = yaml.safe_load(f)

        preProcessing_keys = [k for k in config_yaml if k.startswith("PP")]
        metricmodel_keys = [k for k in config_yaml if k.startswith("MM")]

        exploded_rows = []

        for _, row in raw_input_df.iterrows():
            for ppno in preProcessing_keys:
                for mmno in metricmodel_keys:
                    new_row = deepcopy(row)
                    new_row["payload"] = _parse_payload(new_row["payload"])
                    new_row["job_uuid"] = f"debug_{uuid.uuid4().hex[:12]}"
                    new_row["temp_preProcessingNo"] = ppno
                    new_row["temp_metricModelNo"] = mmno
                    exploded_rows.append(new_row)

        return pd.DataFrame(exploded_rows)





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