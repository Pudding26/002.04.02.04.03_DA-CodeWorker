import pandas as pd
import logging
from typing import Dict
from memory_profiler import profile

from app.tasks.TaskBase import TaskBase
from app.utils.SQL.SQL_Df import SQL_Df
from app.utils.mapping.YamlColumnMapper import YamlColumnMapper


from app.utils.SQL.models.production.api.api_WoodTableA import WoodTableA_Out
from app.utils.SQL.models.production.api.api_WoodTableB import WoodTableB_Out
from app.utils.SQL.models.raw.api.api_PrimaryDataRaw import PrimaryDataRaw_Out

class TA20_B_CreateWoodTableB(TaskBase):
    def setup(self) -> None:
        """Initialize controller message and SQL connection."""
        self.controller.update_message("Initializing Wood Table Creation")
        self.controller.update_progress(0.0)

    def run(self) -> None:
        """Run task logic: load → clean → enrich → filter → store."""
        try:
            self.controller.update_message("Loading required data")
            data_dict = self.load_needed_data()
            self.controller.update_progress(0.2)

            self.controller.update_message("Cleaning and enriching data")
            cleaned_df = self.clean_data(data_dict)
            self.controller.update_progress(0.5)

            self.controller.update_message("Merging with DS09 reference")
            merged_df = self.merge_with_ds09(cleaned_df, data_dict.get("DS09"))
            self.controller.update_progress(0.7)

            self.controller.update_message("Filtering relevant entries")
            final_df = self.filter_data(merged_df)
            self.controller.update_progress(0.9)

            final_df = self.add_woodType(final_df)

            self.controller.update_message("Storing result")

            WoodTableB_Out.store_dataframe(final_df, db_key="production", method="replace")
            logging.info(f"✅ Stored {len(final_df)} rows to WoodTableB")
            self.set_needs_running(False) #mark as already processed for the wrapper

            self.controller.update_progress(1.0)
            self.controller.finalize_success()
        except Exception as e:
            logging.exception("❌ TA20_B_CreateWoodTableB failed.")
            self.controller.finalize_failure(str(e))
            raise
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Flush memory profile logs and archive progress."""
        self.flush_memory_logs()
        self.controller.archive_with_orm()

    def load_needed_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load primaryDataRaw from raw DB, and all other tables from production DB.
        
        Returns:
            dict: Mapping of table name → DataFrame
        """
        df_WoodTableA = WoodTableA_Out.fetch(method="all")

        secured_species_list = df_WoodTableA["species"].dropna().unique().tolist()

        df_PrimaryDataRaw = PrimaryDataRaw_Out.fetch(method="filter", filter_dict={"contains": {
                                                                                          "species": secured_species_list
                                                                                          }})
        

        data_dict = {
            "df_WoodTableA": df_WoodTableA,
            "df_PrimaryDataRaw": df_PrimaryDataRaw,
            "DS09" : SQL_Df("production").load("DS09"),
        }
        
        
        return data_dict


    def clean_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Extracts relevant columns and enriches missing family names in primaryDataRaw.

        Args:
            data_dict: Dict of all loaded tables

        Returns:
            pd.DataFrame: Cleaned and enriched primaryDataRaw
        """
        cols_to_keep = ["family", "IFAW_ID", "genus", "species", "engName", "deName", "frName", "japName", "origin"]
        df_with_fam = pd.DataFrame()



        for name, df in data_dict.items():
            logging.debug3(f"📦 Processing non-raw table: {name}")
            df = df.loc[:, df.columns.intersection(cols_to_keep)]
            if "family" in df.columns:
                fam_subset = df.dropna(subset=["family"])[["family", "genus"]]
                df_with_fam = pd.concat([df_with_fam, fam_subset])


        df_with_fam_dropped = df_with_fam.drop_duplicates()
        logging.debug2(f"🔍 Extracted {len(df_with_fam_dropped)} unique family-genus pairs.")

        df_raw = data_dict["df_PrimaryDataRaw"]
        df_with_family = df_raw[df_raw["family"].notna()].copy()
        df_missing_family = df_raw[df_raw["family"].isna()].copy().drop(columns="family")

        enriched_fam = df_missing_family[["genus"]].drop_duplicates().merge(
            df_with_fam_dropped, on="genus", how="left"
        ).dropna()

        enriched_df = df_missing_family.merge(enriched_fam, on="genus", how="left").dropna(subset=["family"])

        logging.debug2(f"🧬 Enriched {len(enriched_df)} rows with missing family info.")
        final_df = pd.concat([df_with_family, enriched_df])

   

        logging.debug2(f"📊 Cleaned DataFrame total rows: {len(final_df)}")
        return final_df

    def merge_with_ds09(self, merged_df: pd.DataFrame, ds09_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich merged_df using DS09 table based on 'species'.

        Args:
            merged_df: Cleaned DataFrame from clean_data
            ds09_df: Reference table

        Returns:
            pd.DataFrame: Enriched DataFrame
        """
        if ds09_df is None:
            logging.warning("⚠️ DS09 reference table is missing — skipping merge.")
            return merged_df

        ds09_df = ds09_df.dropna(subset=["species"]).copy()
        ds09_df["keep_drop"] = True

        result = merged_df.merge(ds09_df, on="species", how="left", suffixes=('', '_DS09'))
        logging.debug2(f"🔗 Merged with DS09 → {len(result)} rows")

        overlapping = [col for col in ds09_df.columns if col != "species" and col in merged_df.columns]
        logging.debug3(f"🔁 Overriding {len(overlapping)} overlapping columns")

        for col in overlapping:
            override = col + "_DS09"
            if override in result.columns:
                result[col] = result[override].combine_first(result[col])
                result.drop(columns=[override], inplace=True)

        return result

    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters rows by sourceNo and drops temp/drop/old columns.

        Args:
            df: Enriched and merged DataFrame

        Returns:
            pd.DataFrame: Final filtered DataFrame
        """
        logging.debug2(f"Start: Cleaning df for Storage:")
        rouge_cols = ["sourceFilePath_abs", "id"]
        for col in rouge_cols:
            if col in df.columns:
                df.drop(columns=col, inplace=True)
                logging.debug3(f"🗑️ Dropped column '{col}' from DataFrame.")


        df['keep_drop'] = df['sourceNo'].isin(["DS01", "DS04", "DS07", "DS11"])
        filtered = df[df['keep_drop']].drop(columns=["keep_drop"])
        drop_cols = [col for col in filtered.columns if col.endswith(("drop", "temp", "old"))]
        cleaned = filtered.drop(columns=drop_cols, errors='ignore')

        logging.debug2(f"🧹 Filtered down to {len(cleaned)} rows, dropped {len(drop_cols)} columns.")
        return cleaned

    def add_woodType(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds woodType based on family and genus.

        Args:
            df: DataFrame with 'family' and 'genus' columns
        Returns:
            pd.DataFrame: DataFrame with 'woodType' column added
        """
        if "family" not in df.columns or "genus" not in df.columns:
            logging.warning("⚠️ 'family' or 'genus' columns missing — cannot add woodType.")
            return df

        df = df.copy()  # Avoid modifying original DataFrame

        df_mapped = YamlColumnMapper.map_columns_from_yaml(
            df = df,
            new_cols=["woodType"],
            source_cols=["family"],
            yaml_paths=["woodType"],
            yaml_file_path="app/config/mapper/TA20_CreateWoodTable/TA20_0_woodTypeMapper.yaml",
            overwrite=False

        )
        df_mapped["woodType"] = df_mapped["woodType"].replace({
            "Modern Hardwood": "Hardwood",
            "Modern Softwood": "Softwood",
        })
        logging.debug2(f"🌳 Added 'woodType' column with {df['woodType'].nunique()} unique values.")
        return df_mapped
    
