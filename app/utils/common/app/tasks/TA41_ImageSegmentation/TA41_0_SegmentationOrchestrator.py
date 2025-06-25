import logging
import threading
import time
from queue import Queue, Full
from typing import List, Dict
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np
import h5py
import pandas as pd

from threading import Event

from app.tasks.TaskBase import TaskBase

from app.utils.common.app.utils.SQL.models.jobs.api_WorkerJobs import WorkerJobs_Out

from app.utils.common.app.utils.dataModels.Jobs.SegmenterJob import SegmenterJob
from app.utils.common.app.utils.dataModels.Jobs.ExtractorJob import ExtractorJobInput
from app.utils.common.app.utils.dataModels.FilterModel.FilterModel import FilterModel



from app.utils.common.app.utils.dataModels.Jobs.JobEnums import JobStatus


from app.utils.common.app.utils.HDF5.SWMR_HDF5Handler import SWMR_HDF5Handler


from app.tasks.TA41_ImageSegmentation.TA41_A_Segmenter import TA41_A_Segmenter
from app.tasks.TA41_ImageSegmentation.TA41_C_FeatureProcessor import TA41_C_FeatureProcessor



class TA41_0_SegmentationOrchestrator(TaskBase):
    def setup(self):
        logging.debug2("🔧 Running setup...")
        self.jobs = []
        self.controller.update_message("Loading job metadata")


    def run(self):
        try:
            logging.debug2("🚀 Starting main run loop")
            self.controller.update_message("Building job pipeline")
            self.load_jobs_from_db()
            if len (self.jobs) == 0:
                logging.warning("⚠️ No jobs found to process, exiting")
                self.controller.finalize_success()
                return



            num_workers = round(max(1, min(len(self.jobs) // 60, 4))) # returns a minimum of 1 and a maximum of 4 workers
            num_workers = 4
            logging.info(f"🔧 Using {num_workers} worker threads for processing")
            self._run_pipeline(self.jobs, num_loader_workers=num_workers, max_queue_size=50, error_threshold=3)
            self.controller.update_message("Finalizing WoodMaster")
            #TA23_0_CreateWoodMasterPotential.refresh_woodMaster(hdf5_path=self.instructions["HDF5_file_path"])
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

    def load_jobs_from_db(self):
        logging.debug2("📥 Loading SegmenterJobs from database")


        filter_model = FilterModel.from_human_filter({"contains": 
                                                      {
                                                            #"status": "ready", 
                                                            "job_type": "segmenter"
                                                       }})
        
        
        df = WorkerJobs_Out.fetch(filter_model=filter_model)


        total_raw_jobs = len(df)
        self.controller.update_item_count(total_raw_jobs)
        logging.debug2(f"📊 {total_raw_jobs} segmenter job records loaded from DB")

        retry_ready = 0
        retry_delayed = 0
        total_parsed = 0

        for row in df.to_dict(orient="records"):
            try:
                job = SegmenterJob.model_validate(row["payload"])
                total_parsed += 1
                job.job_uuid = row["job_uuid"]
                if job.next_retry <= datetime.now(timezone.utc):
                    self.jobs.append(job)
                    retry_ready += 1
                else:
                    retry_delayed += 1
            except Exception as e:
                logging.error(f"❌ Failed to parse SegmenterJob: {e}", exc_info=True)

        for job_no, job in enumerate(self.jobs):
            job.input.job_No = job_no


        logging.info("📦 Job Loading Summary")
        logging.info(f"  • Total jobs fetched from DB:        {total_raw_jobs}")
        logging.info(f"  • Successfully parsed SegmenterJobs: {total_parsed}")
        logging.info(f"  • Jobs ready to run (retry OK):     {retry_ready}")
        logging.info(f"  • Skipped (next_retry in future):   {retry_delayed}")
        logging.debug3(f"🧱 Built {len(self.jobs)} SegmenterJob objects")


    def _run_pipeline(
        self,
        jobs: List[SegmenterJob],
        num_loader_workers: int = 4,
        max_queue_size: int = 50,
        error_threshold: int = 3,
    ):
        # ✅ Event to block readers until SWMR file is ready
        ready_to_read = Event()

        # ✅ Shared queues
        input_queue: Queue = Queue()
        output_queue: Queue = Queue(maxsize=max_queue_size)
        error_counter = {"count": 0, "lock": threading.Lock()}

        # ✅ Pipeline stats for summary
        pipeline_stats = {
            "total_jobs": len(jobs),
            "total_images": 0,
            "error_count": 0,
            "per_source": defaultdict(lambda: {"jobs": 0, "images": 0, "elapsed": 0.0}),
        }

        def loader(worker_id: int):
            logging.debug(f"[Loader-{worker_id}] Started")
            while True:
                job = input_queue.get()
                if job is None:
                    input_queue.task_done()
                    logging.debug(f"[Loader-{worker_id}] Exiting")
                    break
                if error_counter["count"] >= error_threshold:
                    logging.warning(f"[Loader-{worker_id}] Error threshold exceeded")
                    input_queue.task_done()
                    continue

                try:
                    if job.input.job_No % 25 == 0 or job.input.job_No == 0 or job.input.job_No == len(jobs) - 1: 
                        logging.debug2(f"[Loader-{worker_id}] Processing job #{job.input.job_No} ({job.input.job_No}/{len(jobs)})")

                    # ⏳ Wait until SWMR writer is ready
                    ready_to_read.wait()

                    # ✅ Load image with SWMR reader
                    hdf5 = SWMR_HDF5Handler(job.input.hdf5_path)
                    image_stack = hdf5.load_image(job.input.src_file_path)

                    # ✅ Fetch and prepare attributes
                    attrs_raw = job.attrs.attrs_raw.copy()
                    attrs_raw.pop("parent_job_uuids", None)
                    attrs_raw.update(
                        {"colorDepth": "8bit", "colorSpace": "GS", "filterNo": job.input.dest_FilterNo}
                    )
                    job.attrs.attrs_FF = {**attrs_raw, "filterNo": "FF"}
                    job.attrs.attrs_GS = {**attrs_raw, "filterNo": "GS"}

                    # ✅ Run segmentation
                    segmentor = TA41_A_Segmenter(
                        config=job.input.filter_instructions,
                        image_stack=image_stack,
                        image_stack_id=job.input.dest_stackID_FF,
                        gpu_mode=True,
                    )
                    start_time = time.time()
                    result = segmentor.run_stack()
                    elapsed = time.time() - start_time
                    # ✅ Attach data
                    job.input.image_FF = result.get("filtered_image_stack")
                    job.input.image_GS = result.get("new_gray_stack")
                    job.attrs.segmentation_mask_raw = np.stack(result.get("mask_stack", None))

                    output_queue.put(job)

                    # ✅ Update stats
                    source = job.attrs.attrs_raw.get("sourceNo", "unknown")
                    pipeline_stats["per_source"][source]["jobs"] += 1
                    pipeline_stats["per_source"][source]["images"] += len(job.input.image_FF)
                    pipeline_stats["per_source"][source]["elapsed"] += elapsed
                    pipeline_stats["total_images"] += len(job.input.image_FF)

                except Exception as e:
                    logging.exception(f"[Loader-{worker_id}] Error: {e}")
                    with error_counter["lock"]:
                        error_counter["count"] += 1
                finally:
                    input_queue.task_done()

        def storer():
            logging.debug("[Storer] Started")
            handler = SWMR_HDF5Handler(self.instructions["HDF5_file_path"])

            # ✅ Keep the file open for SWMR writing
            with h5py.File(handler.file_path, "a", libver="latest") as f:
                f.swmr_mode = True
                ready_to_read.set()  # ✅ Signal readers to proceed
                last_job = -1

                while True:
                    job = output_queue.get()
                    if job is None:
                        output_queue.task_done()
                        logging.debug(f"[Storer] Exiting after last job #{last_job}")
                        break

                    if job.input.job_No % 25 == 0 or job.input.job_No == 0 or job.input.job_No == len(jobs) - 1: 
                        logging.debug2(f"[Storer] Processing job #{job.input.job_No} ({job.input.job_No}/{len(jobs)})")


                    last_job = job.input.job_No
                    try:
                        # ✅ Write FF image
                        handler.handle_dataset(
                            hdf5_file=f,
                            dataset_name=job.input.dest_file_path_FF,
                            numpy_array=job.input.image_FF,
                            attributes_new=job.attrs.attrs_FF,
                            attribute_process="att_replace",
                        )
                        job.input.image_FF = None

                        # ✅ Write GS image
                        if job.input.image_GS and not isinstance(job.input.image_GS[0], type(None)):
                            handler.handle_dataset(
                                hdf5_file=f,
                                dataset_name=job.input.dest_file_path_GS,
                                numpy_array=job.input.image_GS,
                                attributes_new=job.attrs.attrs_GS,
                                attribute_process="att_replace",
                            )
                            job.input.image_GS = None

                        f.flush()  # ✅ Flush for readers

                        # ✅ Build ExtractorJobInput
                        if job.attrs.segmentation_mask_raw is not None:
                            mask = job.attrs.segmentation_mask_raw
                            n, h, w = mask.shape
                            job.attrs.extractorJobinput = ExtractorJobInput(
                                mask=mask,
                                n_images=n,
                                width=w,
                                height=h,
                                stackID=job.input.dest_stackID_FF,
                            )
                            job.attrs.segmentation_mask_raw = None

                        




                        # ✅ Save status
                        job.status = JobStatus.IN_PROGRESS.value
                        job.updated = datetime.now(timezone.utc)
                        job.update_db(fields_to_update=["status", "payload"])

                    except Exception as e:
                        logging.exception(f"[Storer] Error on job #{job.input.job_No}: {e}")
                        with error_counter["lock"]:
                            error_counter["count"] += 1
                    finally:
                        output_queue.task_done()

        # ✅ Start storer thread first
        storer_thread = threading.Thread(target=storer, daemon=True)
        storer_thread.start()

        # ✅ Launch loader threads
        loaders = [threading.Thread(target=loader, args=(i,), daemon=True) for i in range(num_loader_workers)]
        for l in loaders:
            l.start()

        # ✅ Feed jobs
        for job in jobs:
            input_queue.put(job)
        if len(jobs) == 0:
            output_queue.put(None)  # No jobs, so stop the storer too

        # ✅ Join
        input_queue.join()
        for _ in loaders:
            input_queue.put(None)
        for l in loaders:
            l.join()

        output_queue.put(None)
        output_queue.join()
        storer_thread.join()

        # ✅ Final summary log
        summary_df = self._create_pipeline_summary(pipeline_stats)
        logging.debug5(
            f"✅ Pipeline completed — "
            f"{len(jobs)} jobs, {pipeline_stats['total_images']} images, errors: {error_counter['count']}"
        )
        logging.debug2("\n" + summary_df.to_string(index=False))


    def _create_pipeline_summary(self, pipeline_stats: Dict) -> pd.DataFrame:
        summary_df = pd.DataFrame.from_dict(pipeline_stats["per_source"], orient="index")
        summary_df.index.name = "sourceNo"
        summary_df.reset_index(inplace=True)

        # Round elapsed time
        summary_df["elapsed"] = summary_df["elapsed"].map(lambda x: round(x, 2))

        # Compute rates
        summary_df["images_per_s"] = (summary_df["images"] / summary_df["elapsed"]).round(2)
        summary_df["stacks_per_s"] = (summary_df["jobs"] / summary_df["elapsed"]).round(2)

        return summary_df



def _safe_store(func, *args, retries: int = 5, delay: float = 0.25, **kwargs):
    """Retry a storage operation that occasionally raises OSError 11 (file lock)."""
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except OSError as e:
            if getattr(e, "errno", None) != 11 or attempt == retries - 1:
                raise
            time.sleep(delay)


