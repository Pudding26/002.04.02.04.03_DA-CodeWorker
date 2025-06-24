import os, io, time, logging
from memory_profiler import profile

from app.tasks.TaskBase import TaskBase
from app.utils.HDF5.HDF5Utils import HDF5Utils


mem_Streams = {
    "step1": io.StringIO(),
    "step2": io.StringIO(),
}



class TA11_B_Import_DS04(TaskBase):
    def setup(self):
        self.dest_filename = self.instructions["destFile_path"]
        self.controller.update_message("Initialized DS04_to_DataRaw04")
        self._profiler_streams = {
            "step1": io.StringIO(),
            "step2": io.StringIO(),
        }

    def run(self):
        try:
            self.controller.update_message("Starting Step 1: File Copy")
            self.step_1_stream_copy()

            self.controller.update_message("Starting Step 2: Attribute Add")
            self.step_2_add_attributes()

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
    def step_1_stream_copy(self):
        src = self.instructions["sourceFile_path"]
        chunk_size = int(self.instructions["chunk_size"]) ** 2
        total_size = os.path.getsize(src)

        copied_size = 0
        progress_update = 0
        update_interval = int(self.instructions["progress_update_cylce"])

        self.controller.update_progress(0.0)
        self.controller.db.set("total_size", total_size)
        self.controller.db.set("copied_size", 0)

        with open(src, "rb", buffering=chunk_size) as fsrc, open(self.dest_filename, "wb") as fdst:
            while True:
                self.check_control()
                chunk = fsrc.read(chunk_size)
                if not chunk:
                    break

                fdst.write(chunk)
                copied_size += len(chunk)
                progress = round(copied_size / total_size, 2)
                self.controller.update_progress(progress)

                if progress * 100 >= progress_update:
                    progress_update += update_interval
                    self.controller.update_progress(progress)
                    self.controller.db.set("copied_size", copied_size)
                    fdst.flush()
                    logging.debug3(f"Progress: {progress}, Copied: {copied_size}")


            os.fsync(fdst.fileno())
            logging.info("File copy complete.")

    @profile(stream=mem_Streams["step2"])
    def step_2_add_attributes(self):
        attributes = {
            "citeKey": self.instructions["citeKey"],
            "sourceNo": self.instructions["sourceNo"]
        }
        result, msg = HDF5Utils.add_attributes_to_hdf5(self.dest_filename, attributes)

        if result != "Finished":
            raise RuntimeError(f"Failed to add attributes: {msg}")

        self.controller.update_message(msg)
        logging.debug5(f"Attributes added: {msg}")
