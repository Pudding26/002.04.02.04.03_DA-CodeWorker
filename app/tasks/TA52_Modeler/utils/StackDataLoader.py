import logging
from typing import List, Dict
import pandas as pd


from app.utils.common.app.utils.dataModels.FilterModel.FilterModel import FilterModel

class StackDataLoader:
    def __init__(self, api_model_cls):
        """
        Parameters:
            api_model_cls: A subclass of api_BaseModel with a .fetch() method.
        """
        self.api_model_cls = api_model_cls
        self._cache: Dict[str, pd.DataFrame] = {}

    def load_for_job(self, stack_ids: List[str]) -> pd.DataFrame:
        """
        Fetches and returns all data related to the given stackIDs.
        Uses internal cache to avoid redundant DB fetches.
        """
        cached_ids = [sid for sid in stack_ids if sid in self._cache]
        uncached_ids = [sid for sid in stack_ids if sid not in self._cache]

        # 1. Serve cached data
        cached_dfs = [self._cache[sid] for sid in cached_ids]

        # 2. Fetch uncached
        if uncached_ids:
            logging.debug2(f"[STACK FETCH] Missing {len(uncached_ids)} stackIDs → fetching from DB.")

            human_filter = {
                "contains": {
                    "stackID": {"or": uncached_ids},
                },
            }
            filter_model = FilterModel.from_human_filter(human_filter)
            



            df_fetched = self.api_model_cls.fetch(filter_model=filter_model, stream=False)

            if df_fetched.empty:
                logging.warning(f"[STACK FETCH] No data returned for stackIDs: {uncached_ids}")
            else:
                for sid in df_fetched["stackID"].unique():
                    df_slice = df_fetched[df_fetched["stackID"] == sid]
                    self._cache[sid] = df_slice
                    cached_dfs.append(df_slice)

                logging.debug2(f"[STACK FETCH] Cached {len(df_fetched)} new rows for {len(uncached_ids)} stackIDs.")

        # 3. Return combined result
        if not cached_dfs:
            return pd.DataFrame()

        return pd.concat(cached_dfs, axis=0, ignore_index=True)



