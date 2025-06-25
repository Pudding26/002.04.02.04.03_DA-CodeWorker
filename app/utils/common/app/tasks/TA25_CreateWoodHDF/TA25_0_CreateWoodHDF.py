import threading
from queue import Queue, Full
import numpy as np
import logging
from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import List, Optional, Dict, Union
import pandas as pd
from datetime import timezone


from threading import Lock
stats_lock = Lock()
from collections import defaultdict
import time

from app.tasks.TaskBase import TaskBase
from app.utils.common.app.utils.controlling.TaskController import TaskController
from app.utils.common.app.utils.HDF5.SWMR_HDF5Handler import SWMR_HDF5Handler

from app.tasks.TA25_CreateWoodHDF._create_stack_and_opt_crop import _create_stack_and_opt_crop

from app.utils.common.app.utils.crawler.Crawler import Crawler

from app.utils.common.app.utils.dataModels.Jobs.util.RetryInfo import RetryInfo
from app.utils.common.app.utils.dataModels.FilterModel.FilterModel import FilterModel

from app.tasks.TA23_CreateWoodMaster.TA23_0_CreateWoodMasterPotential import TA23_0_CreateWoodMasterPotential

from app.utils.common.app.utils.dataModels.Jobs.ProviderJob import ProviderJob
from app.utils.common.app.utils.dataModels.Jobs.JobEnums import JobStatus
from app.utils.common.app.utils.SQL.models.jobs.api_WorkerJobs import WorkerJobs_Out





class WoodJobAttrs(BaseModel):
    Level1: Dict[str, Union[str, int]]
    Level2: Dict[str, Union[str, int]]
    Level3: Dict[str, Union[str, int]]
    Level4: Dict[str, Optional[str]]
    Level5: Dict[str, Union[str, int]]
    Level6: Dict[str, Optional[str]]
    Level7: Dict[str, Union[str, int, float, Optional[str]]]
    dataSet_attrs: Dict[str, Optional[Union[str, float, int]]]

    model_config = ConfigDict(arbitrary_types_allowed=True)

class WoodJobInput(BaseModel):
    src_file_path: str
    src_ds_rel_path: Union[str, List[str]]
    dest_rel_path: str
    image_data: Optional[np.ndarray] = None
    stored_locally: List[int]
    model_config = ConfigDict(arbitrary_types_allowed=True)


class WoodJob(BaseModel):
    jobNo: int
    input: WoodJobInput
    attrs: WoodJobAttrs

    model_config = ConfigDict(arbitrary_types_allowed=True)

class TA25_0_CreateWoodHDF(TaskBase):
    def setup(self):
        logging.debug2("🔧 Running setup...")
        self.jobs = []
        self.controller.update_message("Loading job metadata")
        self._load_jobs_from_db()



    def run(self):
        try:
            logging.debug2("🚀 Starting main run loop")
            if len(self.jobs) ==0:
                sleep_time = self.instructions.get("sleep_time", 1)
                logging.debug5(f"📦 No ProviderJobs found in the database. Sleeping for {sleep_time} s")
                time.sleep(sleep_time)
                return
            self.controller.update_message("Building job pipeline")
            self._run_pipeline(self.jobs)
            self.controller.update_message("Finalizing WoodMaster")
            TA23_0_CreateWoodMasterPotential.refresh_woodMaster(hdf5_path=self.instructions["HDF5_file_path"])
            self.controller.finalize_success()
            logging.info("✅ Task completed successfully")
        except Exception as e:
            self.controller.finalize_failure(str(e))
            logging.error(f"❌ Task failed: {e}", exc_info=True)
            raise
        finally:
            self.cleanup()
            logging.debug2("🧹 Cleanup completed")

    def cleanup(self):
        logging.debug2("🧹 Running cleanup")
        self.flush_memory_logs()
        self.controller.archive_with_orm()

    def _load_jobs_from_db(self):
        logging.debug2("📥 Loading ProviderJobs from database")


        filter_model = FilterModel.from_human_filter({"contains": 
                                                      {"status": "ready", 
                                                       "job_type": "provider"}
                                                      })
        
        
        df = WorkerJobs_Out.fetch(filter_model=filter_model)

        df = df.iloc[:30]
        total_raw_jobs = len(df)
        self.controller.update_item_count(total_raw_jobs)
        logging.debug2(f"📊 {total_raw_jobs} provider job records loaded from DB")


        retry_ready = 0
        retry_delayed = 0
        total_parsed = 0





        for row in df.to_dict(orient="records"):
            try:
                job = ProviderJob.model_validate(row["payload"])
                total_parsed += 1
                job.job_uuid = row["job_uuid"]
                if job.next_retry <= datetime.now(timezone.utc):
                    self.jobs.append(job)
                    retry_ready += 1
                else:
                    retry_delayed += 1
            except Exception as e:
                logging.error(f"❌ Failed to parse ProviderJob: {e}", exc_info=True)

        for job_no, job in enumerate(self.jobs):
            job.input.job_No = job_no


        logging.info("📦 Job Loading Summary")
        logging.info(f"  • Total jobs fetched from DB:        {total_raw_jobs}")
        logging.info(f"  • Successfully parsed ProviderJobs: {total_parsed}")
        logging.info(f"  • Jobs ready to run (retry OK):     {retry_ready}")
        logging.info(f"  • Skipped (next_retry in future):   {retry_delayed}")
        logging.debug3(f"🧱 Built {len(self.jobs)} ProviderJob objects")






    def _run_pipeline(self, jobs: List[WoodJob], num_loader_workers=6, max_queue_size=25, error_threshold=3):
        logging.debug2("🔄 Initializing pipeline queues and threads")
        from threading import Lock
        stats_lock = Lock()

        input_queue = Queue()
        output_queue = Queue(maxsize=max_queue_size)
        error_counter = {"count": 0, "lock": threading.Lock()}



        pipeline_stats = {
            "total_jobs": len(jobs),
            "total_images": 0,
            "total_crops": 0,
            "per_source": defaultdict(lambda: {
                "jobs": 0,
                "images": 0,
                "crops": 0,
                "elapsed": 0.0  # cumulative time
            })
        }


        def loader(worker_id):
            logging.debug2(f"[Loader-{worker_id}] Started")
            print(f"[Loader-{worker_id}] Started")
            while True:
                job = input_queue.get()
                if job is None:
                    logging.debug2(f"[Loader-{worker_id}] Exiting")
                    input_queue.task_done()
                    break
                if error_counter["count"] >= error_threshold:
                    logging.warning(f"[Loader-{worker_id}] Error threshold reached ({error_counter['count']} errors), stopping further processing")
                    #input_queue.task_done()
                    break

                try:
                    self.check_control()


                    job_start_time = time.time()
                    source = job.attrs.Level5["sourceNo"]

                    with stats_lock:
                        stats = pipeline_stats["per_source"][source]
                        stats["jobs"] += 1

                    handler = SWMR_HDF5Handler(file_path=job.input.src_file_path)
                    
                    
                    rel_path = job.input.src_ds_rel_path


                    if job.input.stored_locally[0] == 1: #DIRTY! But allows to later be easier updated so DS11 can be archived locally if latency is terrible
                        handler = SWMR_HDF5Handler(file_path=job.input.src_file_path)
                        if isinstance(rel_path, list):
                            images = [handler.load_image(p) for p in rel_path]
                        else:
                            images = [handler.load_image(rel_path)]
                    else:
                        if isinstance(rel_path, list):
                            images = []
                            for p in rel_path:
                                url = job.input.src_file_path.replace("{id}", p)
                                images.append(Crawler.fetch_image_from_url(url))



                    image_data, was_cropped, filter_type = _create_stack_and_opt_crop(images)
                    


        



                    job.input.image_data = image_data
                    job.attrs.dataSet_attrs.update({
                        "bitDepth": image_data.dtype.itemsize * 8,
                        "colorDepth": "24bit" if filter_type == "RGB" else "8bit",
                        "colorSpace": filter_type,
                        "filterNo": filter_type,
                        "pixel_x": image_data.shape[1],
                        "pixel_y": image_data.shape[2],
                        "was_cropped": was_cropped,
                        
                    })

                    job.input.dest_rel_path += f"/{job.attrs.Level7['sampleID']}_{filter_type}"
                    
                    with stats_lock:
                        pipeline_stats["total_images"] += image_data.shape[0]
                        pipeline_stats["per_source"][source]["images"] += image_data.shape[0]
                        stats["elapsed"] += time.time() - job_start_time

                        if was_cropped:
                            pipeline_stats["total_crops"] += 1
                            pipeline_stats["per_source"][source]["crops"] += 1

                    try:
                        output_queue.put(job, timeout=2)
                    except Full:
                        logging.warning(f"[Loader-{worker_id}] ⏳ Output queue full — job #{job.input.job_No} blocked >2s")
                    logging.debug1(f"[Loader-{worker_id}] Job #{job.input.job_No} prepared — shape {image_data.shape}, type {filter_type}")
                    if job.input.job_No % 10 == 0:
                        logging.debug3(f"[Loader-{worker_id}] Job #{job.input.job_No} prepared — shape {image_data.shape}, type {filter_type}")
                            
                
                except Exception as e:
                    logging.error(f"[Loader-{worker_id}] Error: {e}", exc_info=True)
                    with error_counter["lock"]:
                        error_counter["count"] += 1
                finally:
                    input_queue.task_done()



        def storer():
            handler = SWMR_HDF5Handler(self.instructions["HDF5_file_path"])
            logging.debug2("[Storer] Started")
            last_job = -1
            while True:
                job = output_queue.get()
                if job is not None:
                    last_job = job.input.job_No
                if job is None:
                    logging.debug2(f"[Storer] Exiting, last job was #{last_job}")
                    output_queue.task_done()
                    break
                try:
                    merged_attrs = {**job.attrs.Level1, **job.attrs.Level2, **job.attrs.Level3,
                                    **job.attrs.Level4, **job.attrs.Level5, **job.attrs.Level6,
                                    **job.attrs.Level7, **job.attrs.dataSet_attrs}
                    handler.store_image(dataset_path=job.input.dest_rel_path, image_data=job.input.image_data, attributes=merged_attrs)
                    logging.debug1(f"[Storer] Stored job #{job.input.job_No} → {job.input.dest_rel_path}")
                    if job.input.job_No % 10 == 0:
                        logging.debug3(f"[Storer] Job #{job.input.job_No} stored → {job.input.dest_rel_path}")
                
                    # ✅ Mark job done in DB
                    job.status = JobStatus.DONE
                    job.updated = datetime.now(timezone.utc)
                    
                    job.input.image_data = None
                    job.update_db(fields_to_update=["status"])




                except Exception as e:
                    logging.error(f"[Storer] Error: {e}", exc_info=True)
                    job.register_failure(str(e))
                    if job.attempts >= 5:
                        job.status = JobStatus.FAILED
                    
                    job.input.image_data = None
                    job.update_db(fields_to_update=["status, attempts", "next_retry"])


                    with error_counter["lock"]:
                        error_counter["count"] += 1

                finally:
                    output_queue.task_done()

        s = threading.Thread(target=storer, daemon=True)
        s.start()
        logging.debug2("[Storer] Initialized and started")

        logging.debug3(f"📦 Queuing {len(jobs)} jobs")
        for t in jobs:
            input_queue.put(t)

        logging.debug3(f"Starting storer worker...")

        
        loaders = [threading.Thread(target=loader, args=(i,), daemon=True) for i in range(num_loader_workers)]
        for l in loaders:
            l.start()
            logging.debug2(f"[Loader-{l.name}] Launched")



        input_queue.join()
        for _ in loaders:
            input_queue.put(None)
        for l in loaders:
            l.join()
            logging.debug2(f"[{l.name}] Joined")

        output_queue.put(None)
        output_queue.join()
        s.join()
        logging.debug2("[Storer] Joined")
      
        summary_df = self._create_pipeline_summary(pipeline_stats)

        logging.debug5("📊 Pipeline Summary:")
        logging.debug5(f"🔢 Total Jobs: {pipeline_stats['total_jobs']}")
        logging.debug5(f"🖼️  Total Images: {pipeline_stats['total_images']}")
        logging.debug5(f"✂️  Total Crops: {pipeline_stats['total_crops']}")
        logging.debug5("\n" + summary_df.to_string(index=False))


        logging.info("🎉 Image processing pipeline completed")



    def _create_pipeline_summary(self, pipeline_stats: Dict):
        summary_df = pd.DataFrame.from_dict(pipeline_stats["per_source"], orient="index")
        summary_df.index.name = "sourceNo"
        summary_df.reset_index(inplace=True)
        summary_df["elapsed"] = summary_df["elapsed"].map(lambda x: round(x, 2))
        summary_df["image_per_s"] = (summary_df["images"] / summary_df["elapsed"]).round(2)
        summary_df["stacks_per_s"] = (summary_df["jobs"] / summary_df["elapsed"]).round(2)

        #TA25_0_CreateWoodHDF_stats_OUT.store(

        return summary_df