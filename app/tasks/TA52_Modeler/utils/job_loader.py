import logging
from typing import List, Dict, Any
import pandas as pd
import os


from app.utils.common.app.utils.dataModels.Jobs.ModelerJob import ModelerJob, ModelerJobInput, ModelerAttrs


from app.utils.common.app.utils.SQL.models.production.api_SegmentationResults import SegmentationResults_Out
from app.utils.common.app.utils.SQL.models.production.api_ModellingResults import ModellingResults_Out
from app.utils.common.app.utils.SQL.models.jobs.api_WorkerJobs import WorkerJobs_Out

from app.utils.common.app.utils.dataModels.FilterModel.FilterModel import FilterModel


from app.tasks.TA52_Modeler.utils.StackDataLoader import StackDataLoader
from app.tasks.TA52_Modeler.utils.StackDataLoaderDB import StackDataLoaderDB

from app.tasks.TA52_Modeler.utils.input_queue import input_queue

PERMANENT_CACHE = os.getenv("PERMANENT_CACHE", True)




def load_jobs(job_uuids: list):
    
    filter_model = FilterModel.from_human_filter(
            {"contains": {"job_uuid": job_uuids}},
        )



    job_df = WorkerJobs_Out.fetch(
            filter_model=filter_model,
            stream=False
        )   
    
    job_list = convert_raw_dataframe(job_df)

    if PERMANENT_CACHE == True:
        data_loader = StackDataLoaderDB(api_model_cls=SegmentationResults_Out)
    
    if PERMANENT_CACHE == False:
    
        data_loader = StackDataLoader(api_model_cls=SegmentationResults_Out)

    done_jobs = ModellingResults_Out.fetch_distinct_values(
        column="DoE_UUID",
    )
    already_done = 0
    needs_work = 0
    for i, job in enumerate(job_list):
        try:
            if job.job_uuid in done_jobs:
                logging.debug2(f"[LOADER] Job {job.job_uuid} already processed, skipping.")
                job.status = "DONE"
                job.update_db(fields_to_update=["status", "payload"])
                already_done += 1

            job.attrs.raw_data = data_loader.load_for_job(job.input.stackIDs)

            if i % 10 == 0 or i == len(job_list) - 1:
                logging.debug2(f"[LOADER] Enriched job {i+1}/{len(job_list)} with raw_data")
                logging.debug5(f"[LOADER] Loaded {len(job_list)} jobs, {already_done} already done, {needs_work} need work.")


            input_queue.put(job)
            needs_work += 1 

        

        except Exception as e:
            logging.warning(f"[LOADER WARNING] Failed to attach raw_data to Job {job.job_uuid}: {str(e)}")

    logging.debug5(f"[LOADER] Loaded {len(job_list)} jobs, {already_done} already done, {needs_work} need work.")
    return needs_work, already_done

def convert_raw_dataframe(job_df_raw: pd.DataFrame) -> List[ModelerJob]:
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