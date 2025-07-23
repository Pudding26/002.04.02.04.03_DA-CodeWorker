import sqlite3
import pandas as pd
import logging
import os
from io import StringIO

from typing import List
from filelock import FileLock

from app.utils.common.app.utils.dataModels.FilterModel.FilterModel import FilterModel
from app.utils.common.app.utils.SQL.models.production.api.api_WoodMasterPotential import WoodMasterPotential_Out


from app.tasks.TA52_Modeler.utils.StackDataLoader import StackDataLoader

def resolve_volume_path():
    # Priority order:
    # 1️⃣ Explicit environment variable
    # 2️⃣ Runpod Serverless default
    # 3️⃣ Runpod Pod default
    # 4️⃣ Local fallback (e.g. ./local-cache)

    return (
        os.getenv("RUNPOD_VOLUME_PATH")
        or ("/runpod-volume" if os.path.exists("/runpod-volume") else None)
        or ("runpod-volume" if os.path.exists("runpod-volume") else None)

        or ("/workspace" if os.path.exists("/workspace") else None)
        or "./local-cache"
    )

SQLITE_PATH = os.path.join(resolve_volume_path(), "cache/modeler_wide_cache.db")
logging.info(f"Using SQLite DB path: {SQLITE_PATH}")



class StackDataLoaderDB:
    def __init__(self, api_model_cls, sqlite_path=SQLITE_PATH):
        self.api_model_cls = api_model_cls
        self.sqlite_path = sqlite_path
        self.table_name = "stack_data"
        self.lock = FileLock(f"{self.sqlite_path}.lock")
        self._initialize_db()

    def _initialize_db(self):
        # Ensure directory exists
        parent_dir = os.path.dirname(self.sqlite_path)
        if parent_dir and not os.path.exists(parent_dir):
            logging.info(f"Creating directory for SQLite DB: {parent_dir}")
            os.makedirs(parent_dir, exist_ok=True)

        # Create DB file and initialize schema
        if not os.path.exists(self.sqlite_path):
            logging.info(f"Creating SQLite database at {self.sqlite_path}")
        
        try:
            with self.lock:  # Optional if needed for multi-threaded use
                with sqlite3.connect(self.sqlite_path) as conn:
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self.table_name} (
                            stackID TEXT PRIMARY KEY,
                            data_json TEXT
                        );
                    """)
                    conn.commit()
                    logging.info("✅ SQLite database initialized successfully.")
        except Exception as e:
            logging.error(f"❌ Failed to initialize SQLite DB: {e}")
            raise


    def load_for_job(self, stack_ids: List[str]) -> pd.DataFrame:
        # 1️⃣ Check what’s already cached
        cached_stack_ids = self._get_cached_stack_ids(stack_ids)
        uncached_ids = list(set(stack_ids) - cached_stack_ids)

        logging.debug2(f"[StackDataLoaderDB] Cache hit: {len(cached_stack_ids)}, miss: {len(uncached_ids)}")

        # 2️⃣ Fetch and persist missing
        if uncached_ids:
            df_fetched = self._fetch_and_process(uncached_ids)
            if not df_fetched.empty:
                self._persist_cache(df_fetched)

        # 3️⃣ Read final result for requested stack_ids
        result_df = self._read_cache(stack_ids)
        return result_df

    def _get_cached_stack_ids(self, stack_ids: List[str]) -> set:
        if not stack_ids:
            return set()
        with sqlite3.connect(self.sqlite_path) as conn:
            query = f"""
                SELECT stackID FROM {self.table_name}
                WHERE stackID IN ({','.join(['?'] * len(stack_ids))})
            """
            rows = conn.execute(query, stack_ids).fetchall()
            return set(r[0] for r in rows)

    def _fetch_and_process(self, stack_ids: List[str]) -> pd.DataFrame:
        # This is identical to your existing StackDataLoader logic:
        filter_model = FilterModel.from_human_filter({
            "contains": {"stackID": {"or": stack_ids}}
        })
        df_long = self.api_model_cls.fetch(filter_model=filter_model, stream=False)
        if df_long.empty:
            return pd.DataFrame()

        df_wide = StackDataLoader.reshape_segmentation_long_to_wide(df_long)
        df_wide["sampleID"] = df_wide["stackID"].str.split("_").apply(lambda parts: "_".join(parts[:-1]))
        index_cols = StackDataLoader.get_index_columns(set(df_wide["sampleID"]))

        if index_cols.empty:
            return pd.DataFrame()

        df_wide = pd.merge(df_wide, index_cols, on=["sampleID"], how="left")
        return df_wide

    def _persist_cache(self, df: pd.DataFrame):
        if df.empty:
            return
        # Serialize as JSON row-wise for flexibility (optional optimization: normalize schema)
        records = [
            (row["stackID"], row.to_json())
            for _, row in df.iterrows()
        ]
        with self.lock:
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.executemany(
                    f"INSERT OR REPLACE INTO {self.table_name} (stackID, data_json) VALUES (?, ?)",
                    records
                )
                conn.commit()

    def _read_cache(self, stack_ids: List[str]) -> pd.DataFrame:
        if not stack_ids:
            return pd.DataFrame()
        with sqlite3.connect(self.sqlite_path) as conn:
            query = f"""
                SELECT data_json FROM {self.table_name}
                WHERE stackID IN ({','.join(['?'] * len(stack_ids))})
            """
            rows = conn.execute(query, stack_ids).fetchall()
            records = [pd.read_json(StringIO(r[0]), typ='series') for r in rows]
            return pd.DataFrame(records)




