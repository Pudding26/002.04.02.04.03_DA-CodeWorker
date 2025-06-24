# app/db/sql_df.py

import pandas as pd
import logging
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional, List

from app.utils.SQL.DBEngine import DBEngine
from app.utils.SQL.to_SQLSanitizer import to_SQLSanitizer



class SQL_Df(DBEngine):
    """
    Universal DataFrame-to-SQL interface using a dynamic table name.
    Inherits database engine setup from DBEngine.
    """

    def __init__(self, db_key: str):
        super().__init__(db_key)

    def store(self, table_name: str, df: pd.DataFrame, method: str = "replace") -> None:
        """
        Store a DataFrame in a SQL table.
        method: 'replace' or 'append'
        """
        try:
            
            # Ensure DataFrame is sanitized before storing
            df = df.copy()  # Avoid modifying the original DataFrame
            df = to_SQLSanitizer().sanitize(df)  # Convert to object type and replace fake nulls


            df.to_sql(table_name, self.engine, if_exists=method, index=False)
            logging.debug1(f"Stored DataFrame to table '{table_name}' using method '{method}'.")
        except Exception as e:
            logging.exception(f"❌ Failed to store DataFrame to table '{table_name}': {e}")

    def load(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load a table from SQL into a DataFrame.
        Optionally limit number of rows returned.
        """
        try:
            query = f'SELECT * FROM "{table_name}"'
            if limit:
                query += f" LIMIT {limit}"
            df = pd.read_sql(query, self.engine)
            logging.debug1(f"Loaded {len(df)} rows from table '{table_name}'.")
            return df
        except Exception as e:
            logging.exception(f"❌ Failed to load table '{table_name}': {e}")
            return pd.DataFrame()

    def get_table_names(self) -> List[str]:
        """
        Returns all table names from the current database.
        """
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema='public'
        """
        try:
            with self.get_engine().connect() as conn:
                result = conn.execute(text(query))
                tables = [row[0] for row in result]
            logging.debug1(f"Found tables: {tables}")
            return tables
        except SQLAlchemyError as e:
            logging.exception("❌ Failed to retrieve table names.")
            return []

    def delete_table(self, table_name: str) -> None:
        """
        Deletes the specified SQL table.
        """
        try:
            with self.get_engine().connect() as conn:
                conn.execute(text(f'DROP TABLE IF EXISTS "{table_name}"'))
            logging.debug1(f"🗑️ Deleted table '{table_name}'.")
        except SQLAlchemyError as e:
            logging.exception(f"❌ Failed to delete table '{table_name}': {e}")