import os, io, time, gc, logging
import numpy as np
from PIL import Image
from memory_profiler import profile

from app.tasks.TaskBase import TaskBase
from app.utils.common.app.utils.HDF5.SWMR_HDF5Handler import SWMR_HDF5Handler
from app.utils.common.app.utils.HDF5.HDF5Utils import HDF5Utils

# In-memory memory profiling streams
mem_Streams = {
    "step1": io.StringIO(),
    "step2": io.StringIO(),
    "step3": io.StringIO()
}



class TA11_C_Import_DS07(TaskBase):
    def setup(self):
        self.image_dict = {}
        self.controller.update_message("Initialized DS07_to_DataRaw07")
        self.start_time = time.time()

    def run(self):
        try:
            self.controller.update_message("Starting Step 1")
            self.image_dict = self.step_1_create_image_dict()
            logging.debug5("Step 1 completed, proceeding to Step 2")


            self.controller.update_message("Starting Step 2")
            self.step_2_write_to_hdf5()
            logging.debug5("Step 2 completed, proceeding to Step 3")

            self.controller.update_message("Starting Step 3")
            self.step_3_add_hdf5_attributes()
            logging.debug5("Step 3 completed, finalizing task")

            self.controller.update_progress(1.0)
            self.controller.finalize_success()
            self.set_needs_running(False) #mark as already processed for the wrapper

            self.cleanup()
            logging.debug5("Task completed successfully, cleanup done")
        except Exception as e:
            self.controller.finalize_failure(str(e))
            raise

    def cleanup(self):
        self.flush_memory_logs()
        self.controller.archive_with_orm()

    @profile(stream=mem_Streams["step1"])
    def step_1_create_image_dict(self):
        path = self.instructions["sourceFile_path"]
        image_dict = {}

        if not os.path.isdir(path):
            raise RuntimeError(f"Directory does not exist: {path}")

        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, path)
                    image_dict[file] = {"filepath": full_path, "destpath": rel_path}

        logging.info(f"Found {len(image_dict)} images.")
        return image_dict

    @profile(stream=mem_Streams["step2"])
    def step_2_write_to_hdf5(self):
        dest_file = self.instructions["destFile_path"]
        batch_size = int(self.instructions["batch_size"])

        handler = SWMR_HDF5Handler(dest_file)
        handler.ensure_swmr_hdf5_file()

        total_size = 0
        for root, _, files in os.walk(self.instructions["sourceFile_path"]):
            for file in files:
                total_size += os.path.getsize(os.path.join(root, file))
        self.controller.db.set("total_size", total_size)
        
        update_interval = int(self.instructions["progress_update_cylce"])
        progress_update = 0

        copied_size = 0
        self.controller.db.set("copied_size", 0)

        
        items = list(self.image_dict.items())
        total = len(items)

        self.controller.update_item_count(total)
        self.controller.update_stack_count(total / batch_size)

        for i in range(0, total, batch_size):
            self.check_control()
            batch_items = items[i:i + batch_size]
            for filename, paths in batch_items:
                try:
                    with Image.open(paths["filepath"]) as img:
                        data = np.array(img)

                    handler.store_image(
                        dataset_path=paths["destpath"],
                        image_data=data,
                        attributes={"filename": filename},
                        attribute_process="att_replace"
                    )
                    copied_size += data.nbytes
                    progress = copied_size / total_size



                    if progress * 100 >= progress_update:
                        progress_update += update_interval
                        progress = min(0.95, copied_size / total_size)
                        self.controller.update_progress(progress)
                        roundprogress = round(progress * 100, 2)
                        logging.debug3(f"Progress: {roundprogress}, Copied: {copied_size}")

                    del data
                    gc.collect()
                except Exception as e:
                    logging.warning(f"[SWMR] Failed image {paths['filepath']}: {e}")

            progress = min(0.95, (i + batch_size) / total)
            self.controller.update_progress(progress)

    @profile(stream=mem_Streams["step3"])
    def step_3_add_hdf5_attributes(self):
        logging.debug3("Adding attributes to HDF5 file")
        dest_file = self.instructions["destFile_path"]
        attributes = {
            "citeKey": self.instructions["citeKey"],
            "sourceNo": self.instructions["sourceNo"]
        }
        result, message = HDF5Utils.add_attributes_to_hdf5(dest_file, attributes)

        if result != "Finished":
            raise RuntimeError(f"Failed to add HDF5 attributes: {message}")
