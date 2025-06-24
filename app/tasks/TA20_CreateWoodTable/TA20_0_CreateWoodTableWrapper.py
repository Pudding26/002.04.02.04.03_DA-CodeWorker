import json, yaml
import pandas as pd
import logging

from app.tasks.TaskBase import TaskBase
from app.utils.controlling.TaskController import TaskController
from app.utils.SQL.SQL_Df import SQL_Df
from app.utils.YAML.YAMLUtils import YAMLUtils

from app.utils.SQL.models.production.api.api_DoEArchive import DoEArchive_Out
from app.utils.SQL.models.temp.api.api_DoEJobs import DoEJobs_Out
from app.utils.SQL.models.production.api.api_ModellingResults import ModellingResults_Out




logger = logging.getLogger(__name__)

class TA20_0_CreateWoodTableWrapper(TaskBase):

    def setup(self):
        logger.debug3("🔧 Setting up TA20_0_CreateWoodTableWrapper...")

    def run(self):
        try:
            logging.info("📂 Starting the T1-Tasks")
            self.controller.update_message("📂 Starting the T1-Tasks")
            
            task_list_T1 = self.instructions["tasks_T1"]
            task_list_T1 = self.filter_runnable_tasks(task_list_T1)

            for task_name in task_list_T1:
                self.trigger_task_via_http(task_name=task_name)
            
            
            self.controller.update_message(f"Waiting for the T1-Tasks: {task_list_T1}")
            logging.info(f"📂 Waiting for the T1-Tasks: {task_list_T1}")
            TaskController.watch_task_completion(task_names=task_list_T1, timeout_sec=60, poll_interval=10.0)

            self.controller.update_progress(1.0)
            self.controller.update_message("📂 Cleaning up")

            self.controller.finalize_success()
            logger.info("🎉 TA20_0_CreateWoodTableWrapper completed successfully.")

        except Exception as e:
            logger.error(f"❌ Error during TA20_0_CreateWoodTableWrapper: {e}", exc_info=True)
            self.controller.finalize_failure(str(e))
            raise
        finally:
            self.cleanup()


    def cleanup(self):
        logger.debug3("🧹 Running cleanup phase...")
        self.flush_memory_logs()
        self.controller.archive_with_orm()
        logger.debug3("🧼 Cleanup complete.")

