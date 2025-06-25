import os, io, time, logging, gc
import pandas as pd
from memory_profiler import profile
if os.getenv("DEBUG_MODE") == "True":
    import memory_profiler
    memory_profiler.profile.disable = lambda: None
from app.tasks.TaskBase import TaskBase
from app.utils.common.app.utils.SQL.SQL_Df import SQL_Df
from app.utils.common.app.utils.HDF5.HDF5_Inspector import HDF5Inspector
from app.utils.common.app.utils.mapping.YamlColumnMapper import YamlColumnMapper

mem_Streams = {
    "step1": io.StringIO(),
    "step2": io.StringIO(),
    "step3": io.StringIO()
}

class TA12_C_Transfer_DS07(TaskBase):
    def setup(self):
        logging.info("🔧 Setting up TA12_C_Transfer_DS07")
        self.controller.update_message("Initialized DS07 Transfer")
        self.sql_writer = SQL_Df(db_key=self.instructions["dest_db_name"])
        self.controller.update_progress(0.01)
        logging.debug3(f"🧾 Instructions received: {self.instructions}")

    def run(self):
        try:
            logging.info("🚀 Run started for TA12_C_Transfer_DS07")
            self.controller.update_message("Step 1: Loading HDF5 metadata")
            logging.debug3(f"📂 Opening HDF5 file: {self.instructions['path_DataRaw']}")
            raw_df = self.step_1_load_metadata()
            logging.debug3(f"📊 Loaded HDF5 metadata: shape={raw_df.shape}, columns={list(raw_df.columns)}")
            self.check_control()
            self.controller.update_progress(0.4)

            self.controller.update_message("Step 2: Processing metadata")
            data = self.step_2_preprocess_data(raw_df)
            logging.debug3(f"🧪 Metadata processing complete. Shape: {data.shape}")
            self.check_control()
            self.controller.update_progress(0.7)

            self.controller.update_message("Step 3: Writing to SQL")
            self.step_3_store_to_sql(data)
            self.check_control()

            self.controller.update_progress(1.0)
            self.controller.finalize_success()
            self.set_needs_running(False) #mark as already processed for the wrapper

            logging.info("✅ Task completed successfully")

        except Exception as e:
            logging.exception("❌ Task failed:")
            self.controller.finalize_failure(str(e))
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        logging.debug5("🧹 Running cleanup")
        self.flush_memory_logs()
        self.controller.archive_with_orm()

        logging.debug5("📦 Cleanup and archival complete")

    @profile(stream=mem_Streams["step1"])
    def step_1_load_metadata(self):
        try:
            df = HDF5Inspector.HDF5_meta_to_df(self.instructions["path_DataRaw"])
            logging.info(f"📥 Loaded {len(df)} HDF5 metadata entries")
            return df
        except Exception as e:
            logging.exception("❌ Failed to load HDF5 metadata:")
            raise

    @profile(stream=mem_Streams["step2"])
    def step_2_preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.debug5("🔄 Starting column renaming")
        logging.debug3(f"🪄 Columns before renaming: {list(df.columns)}")

        df = YamlColumnMapper.rename_columns(df, self.instructions["path_gen_col_rename_mapper"]) #NOT FUNCTIONAL
        logging.debug3(f"🪄 Columns after renaming: {list(df.columns)}")

        logging.debug5("🧬 Extracting metadata from path structure")
        df["woodType"] = df["path"].apply(lambda x: x.split("/")[0])
        df["col-1"] = df["path"].apply(lambda x: x.split("/")[1])
        df["col-2"] = df["path"].apply(lambda x: x.split("/")[2])
        df["species"] = df["col-1"].apply(lambda x: ''.join([w.capitalize() for w in x.split(" ")[1:]]))
        df["genus"] = df["col-1"].apply(lambda x: x.split(" ")[1])
        df["specimenNo_old"] = df["col-2"].apply(lambda x: x.split(".")[0][-2:])
        df["source_UUID"] = df["col-2"].apply(lambda x: x.split(".")[0])
        if "path" in df.columns:
            logging.debug2(f"🗂️ Path column found, proceeding with renaming")
            df.rename(columns={"path": "sourceFilePath_rel"}, inplace=True)
        df.drop(columns=["col-1", "col-2"], inplace=True)

        logging.debug5("📥 Adding static YAML columns")
        df = YamlColumnMapper.add_static_columns(df, self.instructions["path_gen_manual_col_mapper"], ["TA12_C_Transfer_DS07"])

        logging.debug5("🧮 Computing image and specimen metadata")
        df["specimenNo"] = df.groupby("species")["source_UUID"].transform(lambda x: pd.factorize(x)[0] + 1)
        df["shotNo"] = 1
        df["totalNumberShots"] = 1
        df["pixel_x"] = df["dataset_shape_drop"].apply(lambda x: x[0] if isinstance(x, tuple) else None)
        df["pixel_y"] = df["dataset_shape_drop"].apply(lambda x: x[1] if isinstance(x, tuple) else None)
        df.drop(columns=["dataset_shape_drop"], inplace=True)
        logging.debug3(f"🧾 Final processed columns: {list(df.columns)}")

        return df

    @profile(stream=mem_Streams["step3"])
    def step_3_store_to_sql(self, df: pd.DataFrame):

        total = len(df)
        logging.info(f"💾 Writing {total} rows to table: {self.instructions['dest_table_name']}")
        try:
            self.sql_writer.store(self.instructions["dest_table_name"], df, method="replace")
            logging.debug2("📤 Data written successfully")
            self.controller.update_message(f"Stored {total} rows into {self.instructions['dest_table_name']}")
        except Exception as e:
            logging.exception("❌ Failed to store data to SQL:")
            raise
