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

class TA11_0_DataImportWrapper(TaskBase):

    def setup(self):
        logger.debug3("🔧 Setting up SQL interface...")
        self.controller.update_message("DoE Task Initialized.")

    def run(self):
        try:
            logger.info("🚀 Starting Datimport Wrapper")
            self.controller.update_message("📂 Starting the tasks")
            
            task_list = self.instructions["tasks_T1"]
            task_list = self.filter_runnable_tasks(task_list)
            for task_name in task_list:
                self.trigger_task_via_http(task_name=task_name)

            TaskController.watch_task_completion(task_list, timeout_sec = 1200)

            
            self.controller.update_progress(1.0)
            self.controller.update_message("📂 Cleaning up")

            self.controller.finalize_success()
            logger.info("🎉 TA11_0_DataImportWrapper completed successfully.")

        except Exception as e:
            logger.error(f"❌ Error during TA11_0_DataImportWrapper: {e}", exc_info=True)
            self.controller.finalize_failure(str(e))
            raise
        finally:
            self.cleanup()


    def cleanup(self):
        logger.debug3("🧹 Running cleanup phase...")
        self.flush_memory_logs()
        self.controller.archive_with_orm()
        logger.debug3("🧼 Cleanup complete.")

