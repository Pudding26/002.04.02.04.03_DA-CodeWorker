import os, io, time, gc, logging, h5py
import numpy as np
from PIL import Image
from memory_profiler import profile




from app.tasks.TaskBase import TaskBase


from app.utils.HDF5.SWMR_HDF5Handler import SWMR_HDF5Handler
from app.utils.HDF5.HDF5Utils import HDF5Utils



# In-memory memory profiling streams
mem_Streams = {
    "step1": io.StringIO(),
    "step2": io.StringIO(),
    "step3": io.StringIO()
}

class TA11_A_Import_DS01(TaskBase):
    def setup(self):
        self.image_dict = {}
        self.controller.update_message("Initialized DS01_to_DataRaw01")
        self.start_time = time.time()

    def run(self):
        try:
            self.controller.update_message("Starting Step 1")
            self.image_dict = self.step_1_create_image_dict()

            self.controller.update_message("Starting Step 2")
            self.step_2_write_to_hdf5()

            self.controller.update_message("Starting Step 3")
            self.step_3_add_hdf5_attributes()

            self.controller.update_progress(1.0)
            self.controller.finalize_success()
            self.set_needs_running(False) #mark as already processed for the wrapper

            self.cleanup()
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
        source_no = self.instructions["sourceNo"]
        ds_name = self.instructions["DS_Name"]
        batch_size = int(self.instructions["batch_size"])


        handler = SWMR_HDF5Handler(self.instructions["destFile_path"])
        handler.ensure_swmr_hdf5_file()  # Ensure it's created if missing

        items = list(self.image_dict.items())
        total = len(items)

        self.controller.update_item_count(total)
        stack_count = total / batch_size
        self.controller.update_stack_count(stack_count)


        logging.debug3(f"Writing {total} images to HDF5 in batches of {batch_size}")
        for i in range(0, total, batch_size):
            self.check_control()

            batch_items = items[i:i + batch_size]
            for filename, paths in batch_items:
                try:
                    with Image.open(paths["filepath"]) as img:
                        data = np.array(img)

                    # Store image using SWMR handler
                    handler.store_image(
                        dataset_path=paths["destpath"],
                        image_data=data,
                        attributes={"filename": filename},
                        attribute_process="att_replace"
                    )

                    del data
                    gc.collect()
                except Exception as e:
                    logging.warning(f"[SWMR] Failed image {paths['filepath']}: {e}")

            progress = min(0.95, (i + batch_size) / total)
            self.controller.update_progress(progress)

    @profile(stream=mem_Streams["step3"])
    def step_3_add_hdf5_attributes(self):
        citeKey = self.instructions["citeKey"]
        sourceNo = self.instructions["sourceNo"]

        if not self.instructions["destFile_path"]:
            raise RuntimeError("Missing file_path for HDF5")

        attributes = {"citeKey": citeKey, "sourceNo": sourceNo}
        result, message = HDF5Utils.add_attributes_to_hdf5(self.instructions["destFile_path"], attributes)

        if result != "Finished":
            raise RuntimeError(f"Failed to add HDF5 attributes: {message}")
