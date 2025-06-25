import logging
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional, List

from app.utils.common.app.utils.SQL.DBEngine import DBEngine


class SQL_Dict(DBEngine):
    """
    Dictionary-style key-value store for SQL.
    Table is auto-created if missing.
    """

    def __init__(self, db_key: str, table_name: str):
        super().__init__(db_key)
        self.table_name = table_name
        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self) -> None:
        """
        Ensures that the key-value table exists.
        """
        query = f"""
        CREATE TABLE IF NOT EXISTS "{self.table_name}" (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
        logging.debug1(f"Eunsuring table in DB: {self.get_engine().url}, table: {self.table_name}")
        try:
            with self.get_engine().begin() as conn:  # ensures transaction is committed
                conn.execute(text(query))

            logging.debug1(f"Ensured key-value table '{self.table_name}' exists.")
        except SQLAlchemyError as e:
            logging.exception(f"❌ Failed to create table '{self.table_name}': {e}")
            raise

    def set(self, key: str, value: str) -> None:
        """
        Sets a value for a key (upsert).
        """
        query = f"""
        INSERT INTO "{self.table_name}" (key, value)
        VALUES (:key, :value)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
        """
        try:
            with self.get_engine().begin() as conn:
                conn.execute(text(query), {"key": key, "value": value})
            logging.debug1(f"Set key '{key}' to value '{value}'.")
        except SQLAlchemyError as e:
            logging.exception(f"❌ Failed to set key '{key}': {e}")

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Retrieves a value for a key, or default if not found.
        """
        query = f'SELECT value FROM "{self.table_name}" WHERE key = :key'
        try:
            with self.get_engine().connect() as conn:
                result = conn.execute(text(query), {"key": key}).fetchone()
            if result:
                logging.debug1(f"Retrieved value for key '{key}'.")
                return result[0]
            return default
        except SQLAlchemyError as e:
            logging.exception(f"❌ Failed to get key '{key}': {e}")
            return default

    def delete(self, key: str) -> None:
        """
        Deletes a key-value pair.
        """
        query = f'DELETE FROM "{self.table_name}" WHERE key = :key'
        try:
            with self.get_engine().begin() as conn:
                conn.execute(text(query), {"key": key})
            logging.debug1(f"Deleted key '{key}'.")
        except SQLAlchemyError as e:
            logging.exception(f"❌ Failed to delete key '{key}': {e}")

    def keys(self) -> List[str]:
        """
        Returns all keys in the table.
        """
        query = f'SELECT key FROM "{self.table_name}"'
        try:
            with self.get_engine().connect() as conn:
                result = conn.execute(text(query))
                keys = [row[0] for row in result]
                logging.debug1(f"Retrieved keys from '{self.table_name}'.")
                return keys
        except SQLAlchemyError as e:
            logging.exception(f"❌ Failed to retrieve keys from '{self.table_name}': {e}")
            return []
