import pandas as pd
import numpy as np

import os
import logging
from app.tasks.TaskBase import TaskBase
from app.utils.common.app.utils.SQL.SQL_Df import SQL_Df

if os.getenv("DEBUG_MODE") == "True":
    import memory_profiler
    memory_profiler.profile.disable = lambda: None
from app.utils.common.app.utils.general.HelperFunctions import generate_deterministic_string_uuid


from app.utils.common.app.utils.SQL.models.raw.orm.PrimaryDataRaw import PrimaryDataRaw
from app.utils.common.app.utils.SQL.models.raw.api.api_PrimaryDataRaw import PrimaryDataRaw_Out



class TA12_E_ConcatPrimaryDataRaw(TaskBase):
    def setup(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("[Setup] Initializing TA12_E_ConcatPrimaryDataRaw task")
        self.controller.update_message("Initializing TA12_E_ConcatPrimaryDataRaw Task")

        self.src_db = SQL_Df(db_key=self.instructions["src_db_name"])
        self.data = pd.DataFrame()

    def run(self):
        try:
            table_list = self.instructions["src_table_names"]
            self.logger.info(f"[Run] Starting concat for tables: {table_list}")
            self.controller.update_item_count(len(table_list))
            self.data_dict = {}

            for idx, table in enumerate(table_list):
                self.check_control()
                self.logger.debug2(f"[Run] Reading table: {table}")
                self.controller.update_message(f"Reading table {table}")
                df = self.src_db.load(table_name=table)
                self.logger.debug3(f"[Run] Loaded {len(df)} rows from {table}")
                self.data_dict[table] = df
                self.data = pd.concat([self.data, df], ignore_index=True)

                progress = (idx + 1) / len(table_list)
                self.controller.update_progress(min(progress, 0.9))




            


            self.controller.update_message("Storing result tables...")
            self.logger.info("[Run] Concatenation complete. Proceeding to store data.")
            self.store_data()

            self.controller.update_progress(1.0)
            self.controller.finalize_success()
            self.set_needs_running(False) #mark as already processed for the wrapper

            self.logger.info("[Run] Task completed successfully.")
        except Exception as e:
            self.logger.error(f"[Run] Concat failed: {e}", exc_info=True)
            self.controller.finalize_failure(str(e))
            raise
        finally:
            self.cleanup()



    def store_data(self):
        

        self.data["raw_UUID"] = "r_" + generate_deterministic_string_uuid(
            self.data["source_UUID"].astype(str).str.cat(self.data["sourceNo"].astype(str), na_rep=""),
            length=6)


        PrimaryDataRaw_Out.store_dataframe(self.data, db_key="raw", method="replace", insert_method = "bulk_save_objects")


        self.logger.info(f"[Store] Data stored via ORM.")

    def cleanup(self):
        self.logger.info("[Cleanup] Flushing memory logs and archiving task progress.")

        self.flush_memory_logs()
        self.controller.archive_with_orm()
        self.logger.info("[Cleanup] Task cleanup completed.")
