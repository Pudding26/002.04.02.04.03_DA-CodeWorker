import logging
import os
import pandas as pd
from typing import List
import yaml
import random

from app.tasks.TaskBase import TaskBase


from app.utils.common.app.utils.SQL.models.temp.api.api_DoEJobs import DoEJobs_Out
from app.utils.common.app.utils.SQL.models.production.api.api_WoodMasterPotential import WoodMasterPotential_Out
from app.utils.common.app.utils.SQL.models.production.api.api_WoodMaster import WoodMaster_Out



from app.utils.common.app.utils.dataModels.FilterModel.FilterModel import FilterModel
from app.utils.common.app.utils.dataModels.FilterModel.FilterModel import Border

from app.utils.common.app.utils.dataModels.Jobs.DoEJob import DoEJob
from app.utils.common.app.utils.dataModels.Jobs.ProviderJob import ProviderJob


#from app.utils.common.app.utils.SQL.models.temp.api.SegmentationJobs_out import SegmentationJobs_out


class TA30_C_ModelerJobBuilder(TaskBase):
    def setup(self):
        self.controller.update_message("Initializing Segmentation Job Builder")
        self.controller.update_progress(0.01)
        self.woodmaster_df = pd.DataFrame()
        self.general_job_df = pd.DataFrame()
        self.filtered_jobs_df = pd.DataFrame()
        self.segmentation_jobs = []
        logging.info("[TA30_A] Setup complete.")

    def run(self):
        try:
            logging.info("[TA30_C] Starting Modeler Job Builder process.")

            pass
        except Exception as e:

            raise
        finally:
            self.cleanup()

    def cleanup(self):
        logging.info("[TA30_A] Running cleanup routine.")
        self.controller.archive_with_orm()