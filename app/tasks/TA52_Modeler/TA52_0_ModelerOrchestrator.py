import logging
import traceback
from queue import Queue
from threading import Thread, Lock
import threading
from typing import List, Dict, Any, Optional
import pandas as pd
import gc


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
        validator = TA52_C_Validator()

        while True:
            job = self.input_queue.get()
            if job is None:
                self.output_queue.put(None)
                self.input_queue.task_done()
                break

            try:
                # Use job as-is — no transformation
                job = preprocessor.run(job)
                if job.status == "FAILED":
                    self._cleanup_job_memory(job)

                    job.update_db(fields_to_update=["status"])
                    continue

                job = modeler.run(job)
                if job.status == "FAILED":
                    self._cleanup_job_memory(job)
                    job.update_db(fields_to_update=["status"])
                    continue
                job = validator.run(job)
                if job.status == "FAILED":
                    self._cleanup_job_memory(job)

                    job.update_db(fields_to_update=["status"])
                    continue

                with self.stats_lock:
                    self.stats_list.append(job)

                    if len(self.stats_list) % 5 == 0:
                        self._print_summary_df(self.stats_list, key="preprocessing")
                        self._print_summary_df(self.stats_list, key="feature_engineering")
                        self._print_summary_df(self.stats_list, key="validation")
                        logging.debug2(f"[STATS] Collected {len(self.stats_list)} stats so far")



                self._cleanup_job_memory(job)
                self.output_queue.put(job)

            except Exception as e:
                logging.exception(f"[{threading.current_thread().name}]")

            finally:
                self.input_queue.task_done()

    def _store_results(self):
        
        @contextmanager
        def suppress_logging(self, level=logging.WARNING):
            """
            Temporarily suppress logging below the given level (default: WARNING).

            Usage:
                with self.suppress_logging():
                    self.fetch(...)
            """
            logger = logging.getLogger()
            previous_level = logger.level
            logger.setLevel(level)
            try:
                yield
            finally:
                logger.setLevel(previous_level)


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
        job.input = None
        job.context = None

        # GPU memory cleanup (CuPy only)
        try:
            import cupy as cp
            cp._default_memory_pool.free_all_blocks()
        except Exception:
            pass  # CuPy not in use or already cleaned

        # Python memory cleanup
        gc.collect()