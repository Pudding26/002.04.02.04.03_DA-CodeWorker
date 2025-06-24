import threading
from queue import Queue, Full
import numpy as np
import logging
from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import List, Optional, Dict, Union
import pandas as pd


from threading import Lock
stats_lock = Lock()
from collections import defaultdict
import time

from app.tasks.TaskBase import TaskBase
from app.utils.controlling.TaskController import TaskController
from app.utils.HDF5.SWMR_HDF5Handler import SWMR_HDF5Handler

from app.tasks.TA25_CreateWoodHDF._create_stack_and_opt_crop import _create_stack_and_opt_crop

from app.utils.crawler.Crawler import Crawler


from app.utils.SQL.models.temp.api.api_PrimaryDataJobs import PrimaryDataJobs_Out

from app.tasks.TA23_CreateWoodMaster.TA23_0_CreateWoodMaster import TA23_0_CreateWoodMaster

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
            self.controller.update_message("Building job pipeline")
            self._run_pipeline(self.jobs)
            self.controller.update_message("Finalizing WoodMaster")
            from app.tasks.TA23_CreateWoodMaster.TA23_0_CreateWoodMaster import TA23_0_CreateWoodMaster
            TA23_0_CreateWoodMaster.refresh_woodMaster(hdf5_path=self.instructions["HDF5_file_path"])
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
        logging.debug2("📥 Loading jobs from database")
        df = PrimaryDataJobs_Out.fetch_all()
        self.controller.update_item_count(len(df))
        logging.debug2(f"📊 {len(df)} records loaded from DB")
        if self.instructions.get("debug", False) == True:
            logging.debug2("🔍 Debug mode enabled, reducing job count for testing")
            df = df[::self.instructions.get("debug_sample_rate", 25)]
            logging.debug2(f"📊 Reduced job count to {len(df)} for debug mode")
        hdf5_path_map = {
            "DS01": "data/rawData/primary/DS01.hdf5",
            "DS04": "data/rawData/primary/DS04.hdf5",
            "DS07": "data/rawData/primary/DS07.hdf5",
            "DS11": "https://iiif-images.lib.ncsu.edu/iiif/2/insidewood-{id}/full/full/0/default.jpg"
        }

        for i, row in enumerate(df.to_dict(orient="records")):
            rel_paths = row["sourceFilePath_rel"]
            if isinstance(rel_paths, str):
                import json
                rel_paths = json.loads(rel_paths)

            job = WoodJob(
                jobNo=i,
                input=WoodJobInput(
                    src_file_path=hdf5_path_map[row["sourceNo"]],
                    src_ds_rel_path=rel_paths,
                    dest_rel_path=row["hdf5_dataset_path"],
                    stored_locally = row["sourceStoredLocally"]
                ),
               attrs = WoodJobAttrs(
                    Level1={
                        "woodType": row.get("woodType"),
                    },
                    Level2={
                        "family": row.get("family"),
                        "genus": row.get("genus"),
                    },
                    Level3={
                        "genus": row.get("genus"),
                    },
                    Level4={
                        "species": row.get("species", "unknown") or "unknown",
                        "engName": row.get("engName", "unknown") or "unknown",
                        "deName": row.get("deName", "unknown") or "unknown",
                        "frName": row.get("frName", "unknown") or "unknown",
                        "japName": row.get("japName", "unknown") or "unknown"
                    },
                    Level5={
                        "sourceID": row["sourceID"],  # preserve ID
                        "sourceNo": row["sourceNo"]
                    },
                    Level6={
                        "specimenID": row["specimenID"],
                        "microscopicTechnic": row.get("microscopicTechnic", "unknown") or "unknown",
                        "institution": row.get("institution", "unknown") or "unknown",
                        "institutionCode": row.get("institutionCode", "unknown") or "unknown",
                        "contributor": row.get("contributor", "unknown") or "unknown",
                        "citeKey": row.get("citeKey", "unknown") or "unknown",
                        "IFAW_code": row.get("IFAW_code", "unknown") or "unknown",
                        "samplingPoint": row.get("samplingPoint", "unknown") or "unknown",
                        "origin": row.get("origin", "unknown") or "unknown"
                    },
                    Level7={
                        "sampleID": row["sampleID"],
                        "digitizedDate": row.get("digitizedDate", "unknown") or "unknown",
                        "view": row.get("view", "unknown") or "unknown",
                        "lens": row.get("lens", "unknown") or "unknown",
                        "totalNumberShots": row.get("totalNumberShots", "unknown") or "unknown"
                    },
                    dataSet_attrs={
                        "bitDepth": None,
                        "colorDepth": None,
                        "colorSpace": None,
                        "pixelSize_um_per_pixel": row.get("pixelSize_um_per_pixel", "unknown") or "unknown",
                        "DPI": row.get("DPI", "unknown") or "unknown",
                        "area_x_mm": row.get("area_x_mm", "unknown") or "unknown",
                        "area_y_mm": row.get("area_y_mm", "unknown") or "unknown",
                        "pixel_x": None,
                        "pixel_y": None,
                        "numericalAperature_NA": row.get("numericalAperature_NA", "unknown") or "unknown",
                        "GPS_Alt": row.get("GPS_Alt", None),
                        "GPS_Lat": row.get("GPS_Lat", None),
                        "GPS_Long": row.get("GPS_Long", None),
                        "digitizedDate": row.get("digitizedDate", None),
                        "raw_UUID": ", ".join(row["raw_UUID"]) if isinstance(row["raw_UUID"], list) else str(row["raw_UUID"])
                    }
                )

            )
            self.jobs.append(job)
        logging.debug3(f"🧱 Built {len(self.jobs)} job objects")


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
                        logging.warning(f"[Loader-{worker_id}] ⏳ Output queue full — job #{job.jobNo} blocked >2s")
                    logging.debug1(f"[Loader-{worker_id}] Job #{job.jobNo} prepared — shape {image_data.shape}, type {filter_type}")
                    if job.jobNo % 10 == 0:
                        logging.debug3(f"[Loader-{worker_id}] Job #{job.jobNo} prepared — shape {image_data.shape}, type {filter_type}")
                            
                
                except Exception as e:
                    logging.error(f"[Loader-{worker_id}] Error: {e}", exc_info=True)
                    with error_counter["lock"]:
                        error_counter["count"] += 1
                finally:
                    input_queue.task_done()


        def storer():
            handler = SWMR_HDF5Handler(self.instructions["HDF5_file_path"])
            logging.debug2("[Storer] Started")
            while True:
                job = output_queue.get()
                if job is not None:
                    last_job = job.jobNo
                if job is None:
                    logging.debug2(f"[Storer] Exiting, last job was #{last_job}")
                    output_queue.task_done()
                    break
                try:
                    merged_attrs = {**job.attrs.Level1, **job.attrs.Level2, **job.attrs.Level3,
                                    **job.attrs.Level4, **job.attrs.Level5, **job.attrs.Level6,
                                    **job.attrs.Level7, **job.attrs.dataSet_attrs}
                    handler.store_image(dataset_path=job.input.dest_rel_path, image_data=job.input.image_data, attributes=merged_attrs)
                    logging.debug1(f"[Storer] Stored job #{job.jobNo} → {job.input.dest_rel_path}")
                    if job.jobNo % 10 == 0:
                        logging.debug3(f"[Storer] Job #{job.jobNo} stored → {job.input.dest_rel_path}")
                except Exception as e:
                    logging.error(f"[Storer] Error: {e}", exc_info=True)
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