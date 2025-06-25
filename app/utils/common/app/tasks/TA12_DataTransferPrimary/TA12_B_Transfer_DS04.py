import os, io, time, logging
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
    "step2": io.StringIO()
}

class TA12_B_Transfer_DS04(TaskBase):
    def setup(self):
        logging.info("🔧 Setting up TA12_B_Transfer_DS04")
        self.controller.update_message("Initialized DS04 Transfer")
        self.sql_writer = SQL_Df(db_key=self.instructions["dest_db_name"])
        self.controller.update_progress(0.01)
        logging.debug3(f"🧾 Instructions received: {self.instructions}")

    def run(self):
        try:
            logging.info("🚀 Run started for TA12_B_Transfer_DS04")
            self.controller.update_message("Step 1: Loading HDF5 metadata")
            logging.debug3(f"📂 Opening HDF5 file: {self.instructions['path_DataRaw']}")
            raw_df = HDF5Inspector.HDF5_meta_to_df(self.instructions["path_DataRaw"])
            logging.debug3(f"📊 Loaded HDF5 metadata: shape={raw_df.shape}, columns={list(raw_df.columns)}")
            self.check_control()
            self.controller.update_progress(0.4)

            self.controller.update_message("Step 2: Processing metadata")
            data = self._process_metadata(raw_df)
            logging.debug3(f"🧪 Metadata processing complete. Shape: {data.shape}")
            
            self.check_control()
            self.controller.update_progress(0.7)
            
            
            self.controller.update_message("Step 3: Writing to SQL")
            self._store_data_with_progress(data)
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
    def _process_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.debug5("🔄 Starting column renaming")
        df = YamlColumnMapper.rename_columns(
            df,
            self.instructions["path_gen_col_rename_mapper"],
            keys_list=["col_names"]
            )        
        logging.debug3(f"🪄 Columns after renaming: {list(df.columns)}")

        if "species_todo" not in df.columns or "family" not in df.columns:
            logging.warning("⚠️ Expected columns 'species_todo' or 'family' missing. Check mapping.")

        logging.debug5("🧬 Formatting taxonomy fields")
        df["species"] = df["species_todo"].apply(lambda x: ''.join(w.capitalize() for w in x.split("_")))
        df["family"] = df["family"].apply(lambda x: ''.join(w.capitalize() for w in x.split()))
        df.drop(columns=["species_todo"], inplace=True)

        logging.debug5("📥 Adding static YAML columns")
        df = YamlColumnMapper.add_static_columns(df, self.instructions["path_gen_manual_col_mapper"], ["TA12_B_Transfer_DS04"])

        logging.debug5("🧮 Computing specimen and image metadata")
        df["specimenNo"] = df.groupby("species")["source_UUID"].transform(lambda x: pd.factorize(x)[0] + 1)
        df["totalNumberShots"] = df["dataset_shape_drop"].apply(lambda x: x[0] if isinstance(x, tuple) else None)
        df["pixel_x"] = df["dataset_shape_drop"].apply(lambda x: x[1] if isinstance(x, tuple) and len(x) > 1 else None)
        df["pixel_y"] = df["dataset_shape_drop"].apply(lambda x: x[2] if isinstance(x, tuple) and len(x) > 2 else None)
        df["DPI"] = round(25400 / df["pixelSize_um_per_pixel"], 2)
        df["sourceFilePath_rel"] = df["source_UUID"]

        df.drop(columns="dataset_shape_drop", inplace=True)
        logging.debug3(f"🧾 Final processed columns: {list(df.columns)}")
        return df

    @profile(stream=mem_Streams["step2"])
    def _store_data_with_progress(self, df: pd.DataFrame):

        
        total = len(df)
        logging.info(f"💾 Storing {total} rows to SQL table: {self.instructions['dest_table_name']}")
        batch_size = 1000
        written = 0

        for i in range(0, total, batch_size):
            self.check_control()
            batch = df.iloc[i:i+batch_size]
            method = "replace" if i == 0 else "append"
            logging.debug3(f"📤 Writing batch {i // batch_size + 1} with {len(batch)} rows")

            self.sql_writer.store(self.instructions["dest_table_name"], batch, method=method)

            written += len(batch)
            progress = round(written / total, 2)
            self.controller.update_progress(min(0.7, progress))

            logging.debug2(f"📈 Progress: {written}/{total} rows written ({int(progress * 100)}%)")
