import os
import logging
import json
import yaml
import pandas as pd
from memory_profiler import profile
if os.getenv("DEBUG_MODE") == "True":
    import memory_profiler
    memory_profiler.profile.disable = lambda: None
from datetime import datetime
from typing import Optional
import numpy as np
from uuid import uuid4

from app.tasks.TaskBase import TaskBase
from app.utils.common.app.utils.SQL.SQL_Df import SQL_Df
from app.utils.common.app.utils.SQL.models.production.api.api_WoodTableA import WoodTableA_Out
from app.utils.common.app.utils.SQL.models.production.api.api_WoodTableB import WoodTableB_Out
from app.utils.common.app.utils.SQL.models.production.api.api_WoodMaster import WoodMaster_Out
from app.utils.common.app.utils.SQL.models.production.api.api_WoodMasterPotential import WoodMasterPotential_Out



from app.utils.common.app.utils.HDF5.HDF5_Inspector import HDF5Inspector



WOOD_MASTER_AGG_CONFIG = {
    "group_first_cols": [
        "woodType", "family", "genus", "species",
        "engName", "deName", "frName", "japName",
        "sourceID", "sourceNo", "specimenID",
        "microscopicTechnic", "institution", "contributor", "digitizedDate",
        "view", "lens", "totalNumberShots", "pixelSize_um_per_pixel",
        "DPI", "hdf5_dataset_path", "samplingPoint", "origin", "institutionCode", "citeKey",
        "numericalAperature_NA", "area_x_mm", "area_y_mm", "IFAW_code",
        "GPS_Alt", "GPS_Lat", "GPS_Long"
    ],
    "group_list_cols": ["sourceFilePath_rel", "source_UUID", "raw_UUID", "sourceStoredLocally"]
}

later_cols = ["filterNo", "colorDepth", "colorSpace", "pixel_x", "pixel_y", "bitDepth"]

class TA23_0_CreateWoodMasterPotential(TaskBase):
    def setup(self):
        self.df_writer = SQL_Df(self.instructions["Thread_progress_db_path"])
        logging.info("Setup complete. SQL writer initialized.")
        self.controller.update_message("Setup complete.")
        self.base_WoodTable = os.getenv("BASE_WOODTABLE", "WoodTableA_Out")
        match self.base_WoodTable:
            case "WoodTableA_Out":
                self.woodTable = WoodTableA_Out
                logging.debug3("Using WoodTableA_Out as base woodTable.")
            case "WoodTableB_Out":
                self.woodTable = WoodTableB_Out
                logging.debug3("Using WoodTableB_Out as base woodTable.")
            case _:
                raise ValueError(f"Unknown woodTable: {self.base_WoodTable}")
         

    def run(self):
        try:
            self.controller.update_message("Loading woodTable...")
            logging.info("🔄 Loading data from WoodTableA_Out...")
            #raw_df = WoodTableA_Out.fetch_all()
            raw_df = self.woodTable.fetch()
            logging.debug2(f"Loaded {len(raw_df)} rows from woodTable.")

            self.controller.update_message("Cleaning woodTable...")
            wood_df = self.clean_woodTable(raw_df)
            logging.debug2(f"Cleaned woodTable: {wood_df.shape[0]} rows, {wood_df.shape[1]} columns.")

            self.controller.update_message("Creating new woodMaster...")
            WoodMasterPotential = self.create_potentialWoodMaster(wood_df)
            logging.debug2(f"New woodMaster created: {WoodMasterPotential.shape}")
            WoodMasterPotential["sampleID_status"] = "todo"
            WoodMasterPotential["transfer_trys"] = 0
            WoodMasterPotential_Out.store_dataframe(WoodMasterPotential, db_key="production", method="replace")


            self.controller.update_progress(1.0)
            self.controller.finalize_success()
        except Exception as e:
            logging.error(f"❌ Task failed: {e}", exc_info=True)
            self.controller.finalize_failure(str(e))
            raise
        finally:
            self.cleanup()


    def cleanup(self):
        logging.info("🧹 Running cleanup and archiving task state.")
        self.flush_memory_logs()
        self.controller.archive_with_orm()

    def clean_woodTable(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.debug3("🔧 Cleaning woodTable data...")

        def sanitize(val):
            return str(val).replace(" ", "_").replace("/", "-")

        def derive_ids(df):
            df["sourceID"] = df["species"] + "_" + df["sourceNo"]
            df["specimenID"] = df["sourceID"] + "_No" + df["specimenNo"].astype(int).astype(str).str.zfill(3)
            df["sampleID"] = df["specimenID"] + "_" + df["view"] + "_x" + df["lens"].astype(str)
            return df



        def add_hdf5_path(df):
            df["hdf5_dataset_path"] = df.apply(lambda row: "/".join([
                sanitize(row["woodType"]), sanitize(row["family"]), sanitize(row["genus"]),
                sanitize(row["species"]), sanitize(row["sourceID"]),
                sanitize(row["specimenID"]), sanitize(row["sampleID"])
            ]), axis=1)
            return df

        df = derive_ids(df)
        df = add_hdf5_path(df)
        logging.debug3("✅ woodTable cleaned.")
        return df

    def create_potentialWoodMaster(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.debug3("🧪 START: Creating the woodMaster_new.")
        logging.debug2(f"Initial woodTable shape: {df.shape}")

        agg_dict = {col: 'first' for col in WOOD_MASTER_AGG_CONFIG["group_first_cols"]}
        agg_dict.update({col: list for col in WOOD_MASTER_AGG_CONFIG["group_list_cols"]})

        result = df.groupby("sampleID", dropna=False).agg(agg_dict).reset_index()
        logging.debug3(f"Grouped woodTable: {result.shape[0]} rows")

        pydantic_order = list(WoodMasterPotential_Out.model_fields.keys())
        reordered_cols = [col for col in pydantic_order if col in result.columns]
        other_cols = [col for col in result.columns if col not in reordered_cols]
        result = result[reordered_cols + other_cols]

        df_to_investigate = result[result['sourceFilePath_rel'].apply(lambda x: len(x) > 1)]
        df_to_investigate_Ids = df_to_investigate["sampleID"].unique()
        df_to_investigate_raw = df[df['sampleID'].isin(df_to_investigate_Ids)]



        if not df_to_investigate_raw.empty:
            logging.debug3(f"Investigating {len(df_to_investigate)} entries with multiple sourceFilePath_rel.")
            debug_df, len_df = self._create_woodMaster_new_debug(df = df_to_investigate_raw, 
                                                                 result = df_to_investigate)



        logging.debug3("✅ END: Created the potentialWoodMaster.")
        return result






    @staticmethod
    def refresh_woodMaster(hdf5_path: str = "data/productionData/primaryData.hdf5") -> pd.DataFrame:
        logging.info(f"🔄 Refreshing woodMaster from: {hdf5_path}")

        if not os.path.exists(hdf5_path):
            logging.warning(f"❌ HDF5 file not found at path: {hdf5_path}")
            return pd.DataFrame(columns=list(WoodMaster_Out.model_fields.keys()))

        shape_old = WoodMaster_Out.db_shape()

        try:
            df = HDF5Inspector.HDF5_meta_to_df(hdf5_path)

            if df.empty:
                logging.warning("⚠️ HDF5 metadata DataFrame is empty.")
                return pd.DataFrame(columns=list(WoodMaster_Out.model_fields.keys()))

            if "dataset_shape_drop" in df.columns:
                df = df.drop(columns=["dataset_shape_drop"])

            df["stackID"] = df["path"].apply(lambda x: x.split("/")[-1])
            #df = TA23_0_CreateWoodMaster._reorder_woodMaster(df)

            WoodMaster_Out.store_dataframe(df, db_key="production", method="replace")
            shape_new = WoodMaster_Out.db_shape()

            logging.debug3(f"✅ Refreshed and stored woodMaster from HDF5. Old shape: {shape_old}, New shape: {shape_new}")
            
            return df

        except Exception as e:
            logging.error(f"❌ Failed to refresh woodMaster: {e}", exc_info=True)


    @staticmethod
    def _reorder_woodMaster(df: pd.DataFrame) -> pd.DataFrame:
        preferred = list(PrimaryDataJobs_Out.model_fields.keys())
        final_cols = preferred + [col for col in df.columns if col not in preferred]
        return df[final_cols]

    def _create_woodMaster_new_debug(self, df: pd.DataFrame, result: pd.DataFrame):
        # ------------------------------------------------------------------
        # 0)  Prep
        # ------------------------------------------------------------------
        KEY = "sampleID"

        # ------------------------------------------------------------------
        # 1)  Fast per-sample COUNTS  (→ len_df)
        # ------------------------------------------------------------------
        len_df = (
            df.groupby(KEY, dropna=False)
            .nunique(dropna=True)           # very fast – C-level
            .reset_index()
        )
        len_df = len_df.replace(0, 1)


        # ------------------------------------------------------------------
        # 2)  Per-sample LISTS of uniques for inspection  (→ debug_lists)
        #     ─────────────────────────────────────────────────────────────
        #     ⚠️  We deliberately DROP the key column before the agg!
        # ------------------------------------------------------------------



        # ------------------------------------------------------------------
        # 3)  Bin the counters  (vectorised, no Python loops over rows)
        # ------------------------------------------------------------------
        bin_config = {
            "ok":     ["sourceFilePath_rel", "raw_UUID", "shotNo", "source_UUID", "digitizedDate"],
            "medium": ["DPI"],
            "bad":    ["lens"],
        }

        all_binned = sum(bin_config.values(), [])          # flat list

        binned_df = pd.DataFrame({KEY: len_df[KEY]})       # start with the key
        for bin_name, cols in bin_config.items():
            binned_df[bin_name] = len_df[cols].sum(axis=1)

        rest_cols = len_df.columns.difference([KEY] + all_binned)
        binned_df["rest"]   = len_df[rest_cols].sum(axis=1)
        binned_df["health"] = binned_df["bad"] + binned_df["rest"]

        # ------------------------------------------------------------------
        # 4)  Final table for debugging / inspection
        # ------------------------------------------------------------------
        df_verbose = df.groupby("sampleID", dropna=False).agg(set).reset_index()
        debug_df = binned_df.merge(df_verbose, on=KEY, how="left")

        return debug_df, len_df



    def legacy(self):
        
        def _refresh_woodMasterJobs(hdf5_path: str, woodMaster) -> pd.DataFrame:
            """
            Refresh the woodMaster jobs from the HDF5 file.
            """
            logging.info(f"🔄 Refreshing woodMaster jobs from: {hdf5_path}")

            sampleIDs = woodMaster["sampleID"].unique()
            PrimaryDataJobs_Out.filter_table_by_dict(
                filter_dict={"sampleID": sampleIDs},
                method="drop"
            )



            self.controller.update_message("Refreshing HDF5 woodMaster...")
            TA23_0_CreateWoodMaster.refresh_woodMaster(self.instructions["HDF5_file_path"])

            self.controller.update_message("Loading old woodMaster...")
            woodMaster_old = WoodMaster_Out.fetch(method="all")
            logging.debug2(f"Old woodMaster loaded: {len(woodMaster_old)} rows.")

            self.controller.update_message("Identifying new sampleIDs...")
            job_df = wood_new[~wood_new["sampleID"].isin(woodMaster_old["sampleID"])]
            logging.info(f"Identified {len(job_df)} new samples.")

            self.controller.update_message("Storing new jobs...")
            job_df = self.prepare_for_sql(job_df)

            logging.debug3(f"Prepared job DataFrame for SQL. Shape: {job_df.shape}")
            PrimaryDataJobs_Out.store_dataframe(job_df, db_key="temp", method="replace")
            logging.info("✅ Stored new job samples in temp DB.")