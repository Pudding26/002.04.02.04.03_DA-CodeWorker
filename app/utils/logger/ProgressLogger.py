# utils/progress/ProgressLogger.py

from sqlalchemy import Table, Column, String, MetaData, create_engine
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.sql import select, insert, update
import os

class ProgressLogger:
    def __init__(self, task_name: str, db_path: str = "sqlite:///thread_progress.db"):
        self.task_name = task_name
        self.table_name = f"progress_{task_name}"
        self.engine = create_engine(db_path)
        self.meta = MetaData()
        self.table = self._get_or_create_table()

    def _get_or_create_table(self):
        # Try to reflect the table; if not exists, create it
        self.meta.reflect(bind=self.engine)
        if self.table_name in self.meta.tables:
            return self.meta.tables[self.table_name]
        else:
            table = Table(
                self.table_name,
                self.meta,
                Column("key", String, primary_key=True),
                Column("value", String),
            )
            table.create(bind=self.engine)
            return table

    def set(self, key: str, value: str):
        with self.engine.connect() as conn:
            stmt = insert(self.table).values(key=key, value=value).on_conflict_do_update(
                index_elements=["key"],
                set_={"value": value}
            )
            conn.execute(stmt)
            conn.commit()

    def get(self, key: str):
        with self.engine.connect() as conn:
            stmt = select(self.table).where(self.table.c.key == key)
            result = conn.execute(stmt).fetchone()
            return result["value"] if result else None

    def all(self):
        with self.engine.connect() as conn:
            stmt = select(self.table)
            result = conn.execute(stmt).fetchall()
            return {row["key"]: row["value"] for row in result}

    def clear(self):
        with self.engine.connect() as conn:
            conn.execute(self.table.delete())
            conn.commit()
