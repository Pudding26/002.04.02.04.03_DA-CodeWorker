import logging
import traceback
from queue import Queue, Empty
from threading import Thread
from typing import List, Dict, Any, Optional
import pandas as pd

from app.tasks.TA52_Modeler.TA52_A_Preprocessor import TA52_A_Preprocessor
from app.tasks.TA52_Modeler.TA52_B_Modeler import TA52_B_Modeler
from app.utils.common.app.utils.dataModels.Jobs.ModelerJob import ModelerJob, ModelerJobInput, ModelerAttrs





from app.utils.common.app.utils.dataModels.FilterModel.FilterModel import FilterModel

from app.utils.common.app.utils.dataModels.Jobs.ModelerJob import ModelerJob
from app.utils.common.app.utils.SQL.models.production.api_SegmentationResults import SegmentationResults_Out


### ---- utils ---- ###
from app.tasks.TA52_Modeler.utils.StackDataLoader import StackDataLoader


class TA52_0_ModelerOrchestrator:
    def run(self, job_df_raw: pd.DataFrame):
        if job_df_raw.empty:
            logging.debug2("[MODEL_ORCH] No jobs found. Exiting orchestrator.")
            return

        logging.info(f"[MODEL_ORCH] Starting with {len(job_df_raw)} jobs.")

        self.raw_input_df = job_df_raw
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.error_queue = Queue()
        self._stop_signal = False

        self.start_pipeline()

    def start_pipeline(self):
        threads = [
            Thread(target=self.load_jobs, name="LoaderThread"),
            Thread(target=self.execute_jobs, name="WorkerThread"),
            Thread(target=self.store_results, name="SaverThread")
        ]


        for t in threads:
            t.start()
        for t in threads:
            t.join()

        logging.info("[MODEL_ORCH] Pipeline completed.")

    def load_jobs(self):
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

        self.input_queue.put(None)

    def execute_jobs(self):
        job_count = 0
        total_jobs = self.input_queue.qsize()
        log_interval = 10

        while True:
            try:
                job = self.input_queue.get(timeout=1)
                if job is None:
                    self.output_queue.put(None)
                    self.input_queue.task_done()
                    break

                try:
                    self.preprocess_job(job)
                    self.model_job(job)
                    self.output_queue.put(job)

                except Exception as e:
                    error_msg = traceback.format_exc()
                    logging.error(f"[ERROR] Job {job.job_uuid} failed:\n{error_msg}")
                    self.error_queue.put((job, e, error_msg))

                job_count += 1
                if job_count % log_interval == 0 or job_count == total_jobs:
                    logging.debug2(
                        f"[WORKER] Processed {job_count}/{total_jobs} jobs"
                    )

                self.input_queue.task_done()

            except Empty:
                continue

    def store_results(self):
        stored_count = 0
        log_interval = 10

        while True:
            try:
                job = self.output_queue.get(timeout=1)
                if job is None:
                    self.output_queue.task_done()
                    break

                try:
                    self.store_job_result(job)
                    stored_count += 1
                    if stored_count % log_interval == 0:
                        logging.debug2(f"[STORER] Stored {stored_count} jobs so far")

                except Exception as e:
                    logging.warning(f"[STORER WARNING] Failed to store Job {job.job_uuid}: {str(e)}")

                self.output_queue.task_done()

            except Empty:
                continue

    def preprocess_job(self, job: ModelerJob):
        preprocessor = TA52_A_Preprocessor(job)
        preprocessor.run()

    def model_job(self, job: ModelerJob):
        modeler = TA52_B_Modeler(job)
        modeler.run()

    def store_job_result(self, job: ModelerJob):
        result_df = job.attrs.model_results
        if result_df is None or result_df.empty:
            logging.warning(f"[STORE WARNING] Empty result for Job {job.job_uuid}")
        else:
            print(f"[STORE] Job {job.job_uuid} → {len(result_df)} results stored.")

    def convert_raw_dataframe(self, job_df_raw: pd.DataFrame) -> List[ModelerJob]:
        jobs = []
        total_jobs = len(job_df_raw)
        log_interval = 10

        for i, (_, row) in enumerate(job_df_raw.iterrows()):
            try:
                payload: Dict[str, Any] = row["payload"]["input"]
                inp = ModelerJobInput(
                    stackIDs=payload["stackIDs"],
                    preprocessing_instructions=payload.get("preprocessing_instructions", {}),
                    metricModel_instructions=payload.get("metricModel_instructions", {}),
                    preProcessingNo=payload["preProcessingNo"],
                    metricModelNo=payload["metricModelNo"]
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
