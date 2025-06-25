import pandas as pd

from app.tasks.TaskBase import TaskBase
from app.utils.common.app.utils.SQL.SQL_Df import SQL_Df
from app.utils.common.app.utils.mapping.YamlColumnMapper import YamlColumnMapper
import logging
from sqlalchemy.orm import Session
from app.utils.common.app.utils.SQL.DBEngine import DBEngine


from app.utils.common.app.utils.SQL.models.production.orm.DS40 import DS40



class TA13_B_DataTransfer_DS40(TaskBase):
    def setup(self):
        self.src_db = SQL_Df(self.instructions["src_db_name"])
        self.table_name = self.instructions["table_name"]
        self.dataset_name = self.instructions["taskName"]
        self.data_raw = None
        self.data_cleaned = None

        logging.debug2(f"[{self.dataset_name}] 🔧 Task setup complete.")
        self.controller.update_message(f"Initialized {self.dataset_name}")

    def run(self):
        try:
            logging.info(f"[{self.dataset_name}] 🚀 Starting DS40 transfer process.")
            self.controller.update_message("Loading data from source DB...")
            self.data_raw = self.src_db.load(self.table_name)
            logging.debug1(f"[{self.dataset_name}] Loaded {len(self.data_raw)} rows.")
            self.controller.update_progress(0.2)

            self.process()
            self.controller.update_progress(0.5)
            self.check_control()

            self.controller.update_message("Storing in destination DB...")
            DS40.store_dataframe(self.data_cleaned, db_key="production", method="replace")

            logging.info(f"[{self.dataset_name}] ✅ Stored cleaned data to production.")
            self.controller.update_progress(1.0)

            self.controller.finalize_success()
            self.set_needs_running(False) #mark as already processed for the wrapper

            logging.info(f"[{self.dataset_name}] 🎉 Task completed successfully.")

        except Exception as e:
            self.controller.finalize_failure(str(e))
            logging.error(f"[{self.dataset_name}] ❌ Task failed: {e}", exc_info=True)
            raise

    def cleanup(self):
        logging.debug2(f"[{self.dataset_name}] 🧹 Cleanup triggered.")

        self.controller.archive_with_orm()

    def process(self):
        try:
            logging.debug1(f"[{self.dataset_name}] 🧪 Processing dataset...")
            self.data_cleaned = self.data_raw.copy()
            logging.debug2(f"[{self.dataset_name}] ✅ Processing complete.")
        except Exception as e:
            logging.error(f"[{self.dataset_name}] ❌ Error in processing: {e}", exc_info=True)
            raise
