import logging
from typing import List
import pandas as pd

from app.utils.common.app.utils.dataModels.FilterModel.FilterModel import FilterModel

from app.utils.common.app.utils.SQL.models.production.api.api_WoodMasterPotential import WoodMasterPotential_Out



class StackDataLoader:
    def __init__(self, api_model_cls):
        """
        Parameters:
            api_model_cls: A subclass of api_BaseModel with a .fetch() method.
        """
        self.api_model_cls = api_model_cls
        self._wide_cache: pd.DataFrame = pd.DataFrame()

    def load_for_job(self, stack_ids: List[str]) -> pd.DataFrame:
        """
        Returns wide-format data for given stackIDs using a global DataFrame cache.
        Fetches and reshapes only uncached stackIDs.
        """
        # Step 1: find what's already cached
        cached_stack_ids = set(self._wide_cache["stackID"]) if not self._wide_cache.empty else set()
        uncached_ids = [sid for sid in stack_ids if sid not in cached_stack_ids]


        if uncached_ids:
            logging.debug2(f"[STACK FETCH] Fetching {len(uncached_ids)} new stackIDs from DB.")

            filter_model = FilterModel.from_human_filter({
                "contains": {
                    "stackID": {"or": uncached_ids}
                }
            })

            df_long = self.api_model_cls.fetch(filter_model=filter_model, stream=False)

            if df_long.empty:
                logging.warning(f"[STACK FETCH] No segmentation data returned for: {uncached_ids}")
            else:
                df_wide = self.reshape_segmentation_long_to_wide(df_long)
                df_wide["sampleID"] = df_wide["stackID"].str.split("_").apply(lambda parts: "_".join(parts[:-1]))
                index_cols = self.get_index_columns(set(df_wide["sampleID"]))

                if index_cols.empty:
                    logging.warning(f"[STACK FETCH] No index columns found for samples: {set(df_wide['sampleID'])}")
                    # Possibly decide how to proceed. You could skip merge or raise exception.
                    # For now, we skip merging:
                    return pd.DataFrame()  # or df_wide without merge, depending on your needs


                df_wide = pd.merge(df_wide, index_cols, on=["sampleID"], how="left")

                
                
                self._wide_cache = (
                    pd.concat([self._wide_cache, df_wide], axis=0, ignore_index=True)
                    .drop_duplicates(subset=["shotID", "stackID"])
                )

                logging.debug2(f"[STACK FETCH] Added {len(df_wide)} wide rows to cache.")

        # Step 2: serve from cache
        result = self._wide_cache.query("stackID in @stack_ids").copy()
        return result


    @staticmethod
    def reshape_segmentation_long_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms long-format segmentation data into wide-format for modeling.

        Includes:
        - Morphological features (e.g., Solidity_area_bin2_mean)
        - Bin count columns (e.g., area_bin2_count)

        Output: one row per (shotID, stackID)
        """
        if df_long.empty:
            return pd.DataFrame()

        # 1. Create feature column names (e.g., Solidity_area_bin2_mean)
        df_long["feature_column"] = (
            df_long["feature_name"].astype(str)
            + "_" + df_long["bin_type"].astype(str)
            + "_" + df_long["bin_label"].astype(str)
            + "_" + df_long["stat_type"].astype(str)
        )

        # Pivot the actual features
        df_features = df_long.pivot_table(
            index=["shotID", "stackID"],
            columns="feature_column",
            values="feature_value",
            aggfunc="first"
        ).reset_index()

        # 2. Extract distinct bin count info per (bin_type, bin_label)
        df_counts_unique = (
            df_long[["shotID", "stackID", "bin_type", "bin_label", "bin_count"]]
            .drop_duplicates(subset=["shotID", "stackID", "bin_type", "bin_label"])
        )

        # Create count and fraction column names
        df_counts_unique["bin_key"] = (
            df_counts_unique["bin_type"].astype(str)
            + "_" + df_counts_unique["bin_label"].astype(str)
        )

        # Compute total count per (shotID, stackID)
        df_counts_unique["bin_count"] = df_counts_unique["bin_count"].fillna(0)
        df_total = (
            df_counts_unique.groupby(["shotID", "stackID"])["bin_count"]
            .sum()
            .reset_index()
            .rename(columns={"bin_count": "bin_total"})
        )

        df_counts_merged = pd.merge(df_counts_unique, df_total, on=["shotID", "stackID"], how="left")
        df_counts_merged["bin_fraction"] = df_counts_merged["bin_count"] / df_counts_merged["bin_total"]

        # Pivot count and fraction columns
        df_counts_merged["count_col"] = df_counts_merged["bin_key"] + "_count"
        df_counts_merged["fraction_col"] = df_counts_merged["bin_key"] + "_fraction"

        df_counts_pivot_count = df_counts_merged.pivot_table(
            index=["shotID", "stackID"],
            columns="count_col",
            values="bin_count",
            aggfunc="first"
        )

        df_counts_pivot_fraction = df_counts_merged.pivot_table(
            index=["shotID", "stackID"],
            columns="fraction_col",
            values="bin_fraction",
            aggfunc="first"
        )

        # Combine count and fraction tables
        df_counts = pd.concat([df_counts_pivot_count, df_counts_pivot_fraction], axis=1).reset_index()



        # 3. Merge feature values with count info
        df_combined = pd.merge(df_features, df_counts, on=["shotID", "stackID"], how="outer")

        # 4. Clean column names
        df_combined.columns.name = None
        df_combined.columns = [str(col) for col in df_combined.columns]

        return df_combined


    @staticmethod
    def get_index_columns(stack_IDs: set) -> pd.DataFrame:
        """
        Fetches index columns for given stack IDs from WoodMasterPotential_Out.
        
        Parameters:
            stack_IDs: Set of stack IDs to filter by.

        Returns:
            DataFrame with index columns: family, genus, species, sourceID, specimenID, sampleID, stackID.
        """
        
        index_cols_list = ["family", "genus", "species", "sourceID", "specimenID", "sampleID"]
        filter_model = FilterModel.from_human_filter({
                "contains": {
                    "sampleID": {"or": stack_IDs}
                }
            })
        index_cols = WoodMasterPotential_Out.fetch(filter_model=filter_model, stream=False)
        index_cols = index_cols[index_cols_list].copy()

        return index_cols





