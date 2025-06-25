import logging, io, os
import pandas as pd
from memory_profiler import profile
if os.getenv("DEBUG_MODE") == "True":
    import memory_profiler
    memory_profiler.profile.disable = lambda: None
from app.tasks.TaskBase import TaskBase
from app.utils.common.app.utils.SQL.SQL_Df import SQL_Df
from app.utils.common.app.utils.HDF5.HDF5_Inspector import HDF5Inspector
from app.utils.common.app.utils.general.HelperFunctions import split_df_based_on_max_split
from app.utils.common.app.utils.mapping.YamlColumnMapper import YamlColumnMapper



mem_Streams = {
    "step2": io.StringIO(),
}

class TA12_A_Transfer_DS01(TaskBase):
    def setup(self):
        logging.info("🔧 Setting up TA12_A_Transfer_DS01")
        self.controller.update_message("Initialized Data Transfer DS01")
        self.sql_writer = SQL_Df(db_key=self.instructions["dest_db_name"])
        self.controller.update_progress(0.01)
        logging.info("🔧 TA12_A_Transfer_DS01: Setup complete")
        self.controller.update_message("Setup complete for Data Transfer DS01")

    def run(self):
        try:
            logging.debug5("🚀 Step 1: Load HDF5 metadata")
            self.controller.update_message("Step 1: Loading HDF5 data")
            raw_df = HDF5Inspector.HDF5_meta_to_df(self.instructions["path_DataRaw"])
            logging.debug3(f"Loaded HDF5 metadata with {len(raw_df)} rows")

            logging.debug5("🔧 Step 2: Preprocess metadata")
            self.controller.update_message("Step 2: Preprocessing metadata")
            data = self._preprocess_metadata(raw_df)

            logging.debug5("➕ Step 3: Add YAML static columns")
            self.controller.update_message("Step 3: Adding static YAML columns")
            data = YamlColumnMapper.add_static_columns(
                data,
                self.instructions["path_gen_manual_col_mapper"],
                ["TA12_A_Transfer_DS01"]
            )

            logging.debug5("🔬 Step 4: Add image dimension metadata")
            self.controller.update_message("Step 4: Finalizing metadata")
            data = self._add_final_metadata(data)

            logging.debug5("💾 Step 5: Store data in SQL")
            self.controller.update_message("Step 5: Writing to SQL")
            self._store_data_with_progress(data)

            self.controller.update_progress(1.0)
            self.controller.finalize_success()
            self.set_needs_running(False) #mark as already processed for the wrapper

        except Exception as e:
            logging.exception("❌ Task failed:")
            self.controller.finalize_failure(str(e))
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        logging.debug5("🧹 Running cleanup and profiling flush")
        self.flush_memory_logs()
        self.controller.archive_with_orm()

    @profile(stream=mem_Streams["step2"])
    def _preprocess_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.debug5("⚙️ Preprocessing columns and extracting species/genus info")
        df.rename(columns={"dasilva_2017": "citeKey", "1": "sourceNo"}, inplace=True)
        df["sourceFilePath_rel"] = df["path"]
        df["species"] = df["path"].apply(lambda x: x.split("/")[0])
        df["genus"] = df["species"].apply(lambda x: x.split("_")[0])
        df["species"] = df["species"].apply(lambda x: ''.join([w.capitalize() for w in x.split("_")]))
        df.rename(columns={"path": "source_UUID"}, inplace=True)

        df["filename_drop"] = df["sourceFilePath_rel"].apply(
            lambda x: "/".join(x.split("/")[1:]) if isinstance(x, str) and "/" in x else x
        )

        split_dict = split_df_based_on_max_split(df, column='filename_drop', separator='_')
        logging.debug3(f"🔍 Split resulted in {len(split_dict)} categories")
        return self._clean_split_dict(split_dict)

    def _clean_split_dict(self, split_dict):
        logging.debug5("🧹 Cleaning and categorizing split groups")
        dropped = {}
        frames = []
        total = sum(len(df) for df in split_dict.values())
        count = 0
        next_log_pct = 10

        for key, df in split_dict.items():
            logging.debug3(f"Processing split key {key} with {len(df)} rows")
            if key in [1, 2, 3, 7] or key > 7:
                dropped[key] = df
                logging.debug3(f"Skipped group {key}")
                continue

            if key in [4, 5, 6]:
                df.rename(columns={"col-0": "specimenID_old"}, inplace=True)
                df["shotNo"] = df.groupby(["species", "specimenID_old"]).cumcount() + 1
                df["totalNumberShots"] = df.groupby(["species", "specimenID_old"])["shotNo"].transform("max")
                df["specimenNo"] = df.groupby("species")["specimenID_old"].transform(lambda x: pd.factorize(x)[0] + 1)
                frames.append(df)

                count += len(df)
                pct_done = int((count / total) * 100)
                if pct_done >= next_log_pct:
                    logging.debug3(f"Cleaning progress: {pct_done}%")
                    next_log_pct += 10

        if frames:
            concat_df = pd.concat(frames, ignore_index=True)
            concat_df = concat_df.loc[:, ~concat_df.columns.str.startswith('col-')]
            concat_df.rename(columns={"max_split": "max_split_drop"}, inplace=True)
            logging.debug3(f"✅ Cleaned data shape: {concat_df.shape}")
            return concat_df
        else:
            logging.error("❌ No usable groups found in split_dict")
            raise RuntimeError("No valid frames found to process.")

    def _add_final_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.debug5("📏 Extracting pixel dimensions from dataset shape")
        df["pixel_x"] = df["dataset_shape_drop"].apply(lambda x: x[0] if isinstance(x, tuple) and len(x) > 1 else None)
        df["pixel_y"] = df["dataset_shape_drop"].apply(lambda x: x[1] if isinstance(x, tuple) and len(x) > 1 else None)
        df.drop(columns="dataset_shape_drop", inplace=True)
        logging.debug3("Pixel metadata added.")
        return df

    def _store_data_with_progress(self, df: pd.DataFrame):
        logging.debug5("🗃️ Starting data write with row progress")
        total_rows = len(df)
        batch_size = 1000
        written = 0
        next_log_pct = 10

        for i in range(0, total_rows, batch_size):
            self.check_control()
            batch = df.iloc[i:i+batch_size]
            if i == 0:
                method = "replace"
            else:
                method = "append"

            self.sql_writer.store(self.instructions["dest_table_name"], batch, method=method)
            written += len(batch)

            progress = round(written / total_rows, 2)
            if progress * 100 >= next_log_pct:
                self.controller.update_progress(progress)
                logging.debug3(f"Writing progress: {int(progress * 100)}%")
                next_log_pct += 10

            for _, row in batch.iterrows():
                logging.debug1(f"Row stored: {row.get('filename_drop', 'unknown')}")
