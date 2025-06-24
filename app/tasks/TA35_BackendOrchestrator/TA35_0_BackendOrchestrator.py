import logging
import json
import yaml
import pprint
import pandas as pd
import httpx

from app.utils.controlling.TaskController import TaskController
from app.tasks.TaskBase import TaskBase
from app.utils.HDF5.SWMR_HDF5Handler import SWMR_HDF5Handler





class TA35_0_BackendOrchestrator(TaskBase):
    def setup(self):
        logging.debug3("🔧 [TA35] Setup started.")
        #self.src_HDF5_inst_1 = SWMR_HDF5Handler(self.instructions["src_db_path_1"])
        #self.src_SQLiteHandler_inst_2 = self.instructions["src_SQLiteHandler"]
        #self.dest_SQLiteHandler_inst_2 = self.instructions["dest_SQLiteHandler"]
        #self.doe_df_raw = pd.DataFrame()
        #self.ml_table_raw = pd.DataFrame()
        #self.doe_df = pd.DataFrame()
        #self.doe_job_list = []
        self.api_base_url = self.instructions.get("api_base_url", "http://localhost:8000")
        self.doe_df_raw = pd.DataFrame()
        self.ml_table_raw = pd.DataFrame()
        logging.debug3("✅ [TA35] Setup complete.")

    def run(self):
        try:
            self.controller.update_message("🔄 Starting DoE pipeline orchestration")

            
            if self.instructions.get("do_import") is True:
                    logging.info("📦 Starting import pipeline (TA11_0_DataImportWrapper)")
                    TaskBase.trigger_task_via_http("TA11_0_DataImportWrapper")
                    TaskController.watch_task_completion(task_names="TA11_0_DataImportWrapper", timeout_sec=1200, poll_interval=10.0)

            if self.instructions.get("do_transfer") is True:
                logging.info("📤 Starting primary and secondary transfer tasks")
                TaskBase.trigger_task_via_http("TA12_0_DataTransferPrimaryWrapper")
                TaskBase.trigger_task_via_http("TA13_0_TransferSecondaryDataWrapper")

            logging.info("🕐 Waiting for TA12_0_DataTransferPrimaryWrapper to complete")
            TaskController.watch_task_completion(task_names="TA12_0_DataTransferPrimaryWrapper", timeout_sec=300, poll_interval=10.0)

            logging.info("🕐 Waiting for TA13_0_TransferSecondaryDataWrapper to complete")
            TaskController.watch_task_completion(task_names="TA13_0_TransferSecondaryDataWrapper", timeout_sec=150, poll_interval=10.0)


            self.trigger_task_via_http("TA20_0_CreateWoodTableWrapper")
            logging.info("🕐 Waiting for TA20_0_CreateWoodTableWrapper to complete")
            TaskController.watch_task_completion(task_names="TA20_0_CreateWoodTableWrapper", timeout_sec=60, poll_interval=10.0)


            self.trigger_task_via_http("TA23_0_CreateWoodMaster")
            logging.info("🕐 Waiting for TA23_0_CreateWoodMaster to complete")
            TaskController.watch_task_completion(task_names="TA23_0_CreateWoodMaster", timeout_sec=30, poll_interval=10.0)



            generalJob_df = TaskBase.trigger_task_via_http("TA27_0_DoEWrapper")
            TaskController.watch_task_completion(task_names="TA27_0_DoEWrapper",  timeout_sec=300, poll_interval=10.0)

            self.create_job_df()
            self.create_job_queue()

            TaskBase.trigger_task_via_http("TA30_B_SegmentationOrchestrator")

            self.controller.update_progress(1.0)
            self.controller.finalize_success()
        except Exception as e:
            logging.exception("❌ [TA35] Pipeline orchestration failed")
            self.controller.finalize_failure(str(e))
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        self.flush_memory_logs()
        self.controller.archive_with_orm()
        logging.debug3("🧼 [TA35] Cleanup complete.")





    def create_job_queue(self):
        self.doe_job_list = []

        for idx, row in self.doe_df.iterrows():
            self.check_control()
            row_dict = dict(row)
            logging.debug3(f"🧩 Rendering job #{idx}: {row_dict}")
            job = self._render_template(row_dict)
            logging.debug3(f"✅ Job created: {job}")
            self.doe_job_list.append(job)

        self.controller.update_item_count(len(self.doe_job_list))
        logging.info(f"✅ Total jobs rendered: {len(self.doe_job_list)}")

    def _render_template(self, row_dict):
        with open("config/templates/DoE_job_template.yaml", "r") as f:
            template = yaml.safe_load(f)

        def fill(node):
            if isinstance(node, dict):
                return {k: fill(v) for k, v in node.items()}
            elif isinstance(node, list):
                return [fill(v) for v in node]
            elif isinstance(node, str) and node.startswith("{") and node.endswith("}"):
                return row_dict.get(node[1:-1], None)
            return node

        job = fill(template)
        job["DoE_UUID"] = row_dict.get("DoE_UUID")
        return job

    def _load_table(self, handler, table_name):
        try:
            return handler.get_complete_Dataframe(table_name=table_name)
        except Exception as e:
            logging.warning(f"⚠️ Failed to load table {table_name}: {e}")
            return pd.DataFrame()
