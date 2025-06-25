import json, yaml
import pandas as pd
import logging

from app.tasks.TaskBase import TaskBase
from app.tasks.TA27_DesignOfExperiments.TA27_B_DoEExpander import TA27_B_DoEExpander
from app.tasks.TA27_DesignOfExperiments.TA27_A_DoEJobGenerator import TA27_A_DoEJobGenerator
from app.utils.common.app.utils.SQL.SQL_Df import SQL_Df
from app.utils.common.app.utils.YAML.YAMLUtils import YAMLUtils

from app.utils.common.app.utils.SQL.models.production.api.api_DoEArchive import DoEArchive_Out
from app.utils.common.app.utils.SQL.models.jobs.api_DoEJobs import DoEJobs_Out 
from app.utils.common.app.utils.SQL.models.production.api.api_ModellingResults import ModellingResults_Out


logger = logging.getLogger(__name__)

class TA27_0_DoEWrapper(TaskBase):

    def setup(self):
        logger.debug3("🔧 Setting up SQL interface...")
        self.sql = SQL_Df(self.instructions["dest_db_path_1"])
        self.controller.update_message("DoE Task Initialized.")
        self.job_df_clean = pd.DataFrame()

    def run(self):
        try:
            logger.info("🚀 Starting DoE task")
            self.controller.update_message("📂 Loading DoE YAML")
            logger.debug3(f"📥 Loading DoE YAML from: {self.instructions['doe_yaml_path']}")
            raw_yaml = YAMLUtils.load_yaml(self.instructions["doe_yaml_path"])


            logger.debug3("🧮 Expanding parameter combinations...")
            self.doe_df_raw = TA27_B_DoEExpander.expand(raw_yaml)
            logger.debug3(f"🔢 Expanded DoE to {len(self.doe_df_raw)} combinations")

            self.controller.update_message("🧪 Filtering new DoEJobs agains ModellingResults SQL")
            logger.debug3("🔍 Filtering new DoEJobs against ModellingResults...")
            self.create_jobs()



            
            self.controller.update_message("🧪 Saving new DoEJobs to SQL")
            
            
            if self.job_df_clean.empty is False:
                logger.debug3(f"💾 Storing {len(self.job_df_clean)} new DoEJobs in SQL...")
                DoEJobs_Out.store_dataframe(self.job_df_clean, method="append")
                logger.info("✅ New DoE table stored")
            else:
                logging.debug3("⚠️ No new DoE jobs to store in SQL. Skipping.")

            





            #self.controller.update_message("🛠 Generating job definitions")
            #logger.debug3("🧬 Starting job generation...")
            #jobs = TA27_A_DoEJobGenerator.generate(df, self.instructions["job_template_path"])
            #logger.debug3(f"📦 Generated {len(jobs)} job definitions")
            
           

            self.controller.update_progress(1.0)
            self.controller.finalize_success()
            logger.info("🎉 DoE task completed successfully.")

        except Exception as e:
            logger.error(f"❌ Error during DoE task: {e}", exc_info=True)
            self.controller.finalize_failure(str(e))
            raise
        finally:
            self.cleanup()



    def create_jobs(self):


        old_doe_uuids = DoEJobs_Out.fetch_distinct_values("job_uuid")
        logging.debug2(f"✅ Found {len(old_doe_uuids)} old DoE UUIDs.")
        

        self.doe_df = self.doe_df_raw[~self.doe_df_raw["DoE_UUID"].isin(old_doe_uuids)]

        logging.debug5(f"✅ Reduced to {len(self.doe_df)} DoE jobs.")
        if self.doe_df.empty:
            logger.warning("⚠️ No new DoE jobs to process. Exiting.")
            
            return
        

        self.jobs_list = TA27_A_DoEJobGenerator.generate(df = self.doe_df, template_path = self.instructions["job_template_path"])
        
        
        self.job_df_clean = pd.DataFrame([job.to_sql_row() for job in self.jobs_list])




    def cleanup(self):
        logger.debug3("🧹 Running cleanup phase...")
        self.flush_memory_logs()
        self.controller.archive_with_orm()
        logger.debug3("🧼 Cleanup complete.")

