import logging
import os
import pandas as pd
from typing import List
import yaml
import random

from app.tasks.TaskBase import TaskBase
from app.utils.SQL.SQL_Df import SQL_Df
from app.utils.SQL.SQL_Dict import SQL_Dict
from app.utils.SQL.models.temp.api.api_DoEJobs import DoEJobs_Out
from app.utils.SQL.models.production.api.api_WoodTableA import WoodTableA_Out
from app.utils.SQL.models.production.api.api_WoodTableB import WoodTableB_Out
from app.utils.SQL.models.production.api.api_WoodMaster import WoodMaster_Out

from app.utils.dataModels.FilterModel.FilterModel import FilterModel
from app.utils.dataModels.FilterModel.FilterModel import Border



#from app.utils.SQL.models.temp.api.SegmentationJobs_out import SegmentationJobs_out


class TA29_A_SegmentationJobBuilder(TaskBase):
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
            self.controller.update_message("Loading DoE Jobs")
            logging.info("[TA30_A] Fetching DoE jobs via Pydantic model.")
            self.load_general_job_queue()

            self.controller.update_message("Filtering and preparing segmentation jobs")
            logging.info("[TA30_A] Filtering jobs based on DoE definitions.")
            self.filtered_jobs_df = self.filter_jobs()
            self.filtered_jobs_df = self.add_Segmentationfilter_configurations(self.filtered_jobs_df)
            self.segmentation_jobs = self.create_segmentation_jobs(self.filtered_jobs_df)

            self.controller.update_message("Storing jobs to segmentationJobs_out")
            logging.info("[TA30_A] Storing segmentation job DataFrame via Pydantic model.")
            segmentationJobs_out.store_Dataframe(pd.DataFrame(self.segmentation_jobs))

            self.controller.update_progress(1.0)
            self.controller.finalize_success()
            logging.info("[TA30_A] Task completed successfully.")
        except Exception as e:
            self.controller.finalize_failure(str(e))
            logging.exception("[TA30_A] Task failed with exception:", exc_info=True)
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        logging.info("[TA30_A] Running cleanup routine.")
        self.controller.archive_with_orm()

    def load_general_job_queue(self):
        # Use DoEJobs_Out to fetch generalJobQueue
        logging.debug3("[TA30_A] Loading generalJobQueue from DoEJobs_Out model.")
        self.general_job_df = DoEJobs_Out.fetch(method="all")
        if self.general_job_df.empty:
            raise RuntimeError("No DoE jobs found.")
        logging.debug3(f"[TA30_A] Loaded {len(self.general_job_df)} DoE job rows.")

    def filter_jobs(self) -> pd.DataFrame:


        def _job_row_to_filter_model(
            row: pd.Series,
            include_cols: list[str],
            is_range_cols: list[str] = None,
            is_max: list[str] = None,
            is_min: list[str] = None,
            border_rule: dict = None,
            global_logic: str = "and"
        ) -> FilterModel:
            is_range_cols = is_range_cols or []
            is_max = is_max or []
            is_min = is_min or []
            border_rule = border_rule or {}

            return FilterModel.from_row(
                row=row,
                include_cols=include_cols,
                range_cols=is_range_cols,
                min_cols=is_min,
                max_cols=is_max,
                border_rule={k: Border(border_rule[k]) for k in border_rule},
                global_logic=global_logic,
            )

        def _extract_primary_keys_from_yaml(yaml_path: str) -> list[str]:
            with open(yaml_path, "r") as f:
                config = yaml.safe_load(f)
            primary_data = config.get("primary_data", {})
            return list(primary_data.keys())

        include_cols = _extract_primary_keys_from_yaml(self.instructions["DoE_job_template_path"])
        if "maxShots" in include_cols:
            include_cols.remove("maxShots")

        if "totalNumberShots" not in include_cols:
            include_cols.append("totalNumberShots")

        self.general_job_df.rename(columns={"noShotsRange": "totalNumberShots"}, inplace=True)

        job_filters = [
            _job_row_to_filter_model(
                row,
                include_cols=include_cols,
                is_range_cols=["totalNumberShots"],
                is_max=["totalNumberShots"],
                global_logic="and"
            )
            for _, row in self.general_job_df.iterrows()
        ]

        segmentationJobs_df = pd.DataFrame()
        total_jobs = len(job_filters)

        #job_filters = job_filters[:3]  # Limit to first 100 jobs for testing


        for i, filter_model in enumerate(job_filters):
            #if not filter_model.has_conditions():
            #    logging.warning(f"[TA30_A] Job {i} has no valid filter conditions, skipping.")
            #    continue

            if i % 10 == 0 or i == total_jobs - 1:
                logging.debug2(f"[TA30_A] Processing job {i}/{total_jobs}")

            with self.suppress_logging():
                new_subset = WoodTableB_Out.fetch(
                    filter_model=filter_model,
                    stream=False
                )

            len_new_subset = len(new_subset)
            len_old_subset_before = len(segmentationJobs_df)

            if not new_subset.empty:
                segmentationJobs_df = pd.concat([segmentationJobs_df, new_subset], ignore_index=True)

            len_old_subset_after = len(segmentationJobs_df)
            len_delta = len_old_subset_after - len_old_subset_before

            if i % 10 == 0 or i == total_jobs - 1:
                logging.debug2(f"[TA30_A] Job {i}/{total_jobs} Old subset length: {len_old_subset_before}, New subset length: {len_new_subset}, Combined length: {len_old_subset_after}. Ratio added {len_delta / len_new_subset if len_new_subset > 0 else 'N/A'}")

        return segmentationJobs_df




            



        filter_sets = self._extract_filter_sets()
        
        logging.debug3("[TA30_A] Starting job filtering.")

        logging.debug2("[TA30_A] Loading WoodTableA via Pydantic model.")
        woodTable_df = WoodTableA_Out.fetch(method="all")
        if woodTable_df.empty:
            raise RuntimeError("woodTable_df table is empty or missing.")
        logging.debug2(f"[TA30_A] Loaded wood table with {len(woodTable_df)} rows.")

        all_conditions = []
        for job in self.general_job_df.to_dict(orient="records"):
            conditions = []
            for key, values in job.items():
                if isinstance(values, list) and key in woodTable_df.columns:
                    conditions.append(woodTable_df[key].isin(values))
            if conditions:
                combined = conditions[0]
                for cond in conditions[1:]:
                    combined |= cond  # OR filtering
                all_conditions.append(combined)

        if not all_conditions:
            raise RuntimeError("No filter conditions constructed from DoE jobs.")
        
        final_filter = all_conditions[0]
        for cond in all_conditions[1:]:
            final_filter |= cond

        jobs_df = woodTable_df[final_filter].copy()
        if jobs_df.empty:
            raise RuntimeError("No stackIDs matched the provided filter conditions.")

        logging.debug2(f"[TA30_A] Filtered down to {len(jobs_df)} rows after OR filtering.")

        jobs_df["dest_filterNo"] = jobs_df["filterNo"]  # If available
        jobs_df["dest_stackID"] = jobs_df.apply(
            lambda row: row["stackID"].rsplit("_", 1)[0] + f"_{row['dest_filterNo']}", axis=1
        )

        logging.debug3(f"[TA30_A] Job filtering complete. Total: {len(jobs_df)}")
        return jobs_df


    def add_Segmentationfilter_configurations(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.debug("[TA30_A] Adding Segmentationfilter configurations to job table.")
        import yaml
        with open(self.instructions["segmentationFilter_config_path"], "r") as file:
            config = yaml.safe_load(file)
        df["filter_config"] = df["dest_filterNo"].map(config)
        return df

    def create_segmentation_jobs(self, df: pd.DataFrame) -> List[dict]:
        logging.debug("[TA30_A] Creating job dicts from DataFrame.")
        jobs = []
        for _, row in df.iterrows():
            dest_stackID = row["dest_stackID"]
            stack_dir = os.path.dirname(row["path"])
            jobs.append({
                "src_path": row["path"],
                "filterNo": row["dest_filterNo"],
                "dest_path": os.path.join(stack_dir, dest_stackID),
                "dest_stackID": dest_stackID,
                "hdf5_path": self.instructions["hdf5_output_path"],
                "feature_storage_db_path": self.instructions["segmentation_db"],
                "feature_storage_table": self.instructions["segmentation_table"],
                "filter_config": row["filter_config"],
            })
        logging.info(f"[TA30_A] Created {len(jobs)} segmentation job entries.")
        return jobs



### Helper

    def _extract_filter_sets(self) -> dict:
        logging.debug3("[TA30_A] Extracting grouped filter value sets from general_job_df.")
        filter_sets = {}

        for job in self.general_job_df.to_dict(orient="records"):
            for key, values in job.items():
                if isinstance(values, list):
                    if key not in filter_sets:
                        filter_sets[key] = set()
                    # Treat nested lists as atomic units
                    for v in values:
                        try:
                            filter_sets[key].add(tuple(v) if isinstance(v, list) else v)
                        except TypeError:
                            logging.debug1(f"[TA30_A] Skipping unhashable value in {key}: {v}")

        logging.debug2(f"[TA30_A] Extracted filter keys: {list(filter_sets.keys())}")
        return filter_sets
