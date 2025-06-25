import logging
import time
import uuid
from datetime import datetime
from typing import Optional, Union, List
import time

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.utils.common.app.utils.SQL.models.progress.api.api_progressArchive import ProgressArchiveOut
from app.utils.common.app.utils.SQL.SQL_Dict import SQL_Dict  # adjust if your path differs
from app.utils.common.app.utils.SQL.DBEngine import DBEngine  # adjust if your path differs

class TaskController:
    def __init__(self, task_name: str, db_key: str = "progress", task_uuid: str = None):
        """
        db_key: name of your database (e.g., 'progressdb')
        task_name: table name (1 per task, based on task_name)
        """
        self.db = SQL_Dict(db_key=db_key, table_name=task_name)
        self.task_uuid = task_uuid or str(uuid.uuid4())
        self.task_name = task_name
        #self._ensure_task_uuid()

        self.db.set("start_time", str(datetime.now()))


    def _ensure_task_uuid(self):
        if not self.db.get("task_uuid"):
            self.db.set("task_uuid", self.task_uuid)


    def request_stop(self):
        """
        Signals the task to stop by setting the 'Stop' flag in the DB.
        """
        self.db.set("Stop", "1")
        self.db.set("stop_requested_at", str(datetime.now()))
        logging.debug2(f"🛑 Stop requested for task '{self.db.table_name}' (UUID: {self.task_uuid})")


    def should_stop(self):
        return self.db.get("Stop", "0") == "1"

    def should_pause(self):
        return self.db.get("Pause", "0") == "1"

    def is_finished(self):
        return self.db.get("Finished", "0") == "1"

    def get_item_count(self):
        return self._safe_int("item_count")

    def get_stack_count(self):
        return self._safe_int("stack_count")

    def update_progress(self, value: float):
        self.db.set("progress", str(value))

    def update_message(self, msg: str):
        self.db.set("message", msg)

    def update_item_count(self, count: int):
        self.db.set("item_count", str(count))

    def update_stack_count(self, count: int):
        self.db.set("stack_count", str(count))

    def finalize_success(self):
        self.db.set("Finished", "1")
        self.db.set("Status", "Ready")
        self.db.set("message", "Task completed successfully")

    def finalize_failure(self, error_msg: str):
        self.db.set("Status", "Failed")
        self.db.set("message", error_msg)


    def wait_if_paused(self):
        if self.should_pause():
            self.db.set("Status", "Paused")
            while self.should_pause():
                time.sleep(0.5)
            self.db.set("Status", "Running")


    def archive_with_orm(self, force_replace: bool = False):
        """Saves progress keys to ORM table before dropping. Can force overwrite in archive."""

        logging.info(
            f"📦 Archiving task '{self.task_name}' with UUID {self.task_uuid} "
            f"(status: {self.db.get('Status')}, force_replace={force_replace})"
        )
        if getattr(self, "_archived", False):
            logging.debug2(f"📦 Archive already completed for task '{self.task_name}' (UUID: {self.task_uuid}). Skipping.")
            return
        self._archived = True

        now = datetime.now()
        start_dt = self._safe_datetime("start_time")
        elapsed_time = (now - start_dt).total_seconds() if start_dt else 0
        data_transferred_gb = self._safe_float("total_size") / (1024 ** 3) if self._safe_float("total_size") else 0

        archive_model = ProgressArchiveOut(
            task_uuid=self.task_uuid,
            task_name=self.task_name,
            start_time=start_dt,
            finish_time=now,
            status=self.db.get("Status"),
            finished=self.db.get("Finished"),
            message=self.db.get("message"),
            progress=self._safe_float("progress"),
            elapsed_time=elapsed_time,
            total_size=self._safe_float("total_size"),
            data_transferred_gb=data_transferred_gb,
            item_count=self._safe_int("item_count"),
            stack_count=self._safe_int("stack_count"),
        )


        try:
            archive_model.persist_to_db(archive_model, force_replace=force_replace)
            logging.debug2(f"✅ Archive persisted for task '{self.task_name}' (UUID: {self.task_uuid})")
        except Exception as e:
            logging.error(f"❌ Archive failed for task '{self.task_name}' (UUID: {self.task_uuid}): {e}", exc_info=True)
            return

        try:
            with self.db.get_engine().begin() as conn:
                conn.execute(text(f'DROP TABLE IF EXISTS \"{self.db.table_name}\"'))
                logging.debug2(f"🧹 Dropped progress table '{self.db.table_name}' after archival.")
        except Exception as e:
            logging.error(f"❌ Failed to drop progress table '{self.db.table_name}': {e}", exc_info=True)


    def _safe_int(self, key: str) -> int:
        try:
            return int(self.db.get(key) or 0)
        except (ValueError, TypeError):
            return 0

    def _safe_float(self, key: str) -> float:
        try:
            return float(self.db.get(key) or 0.0)
        except (ValueError, TypeError):
            return 0.0

    def _safe_datetime(self, key: str) -> Optional[datetime]:
        val = self.db.get(key)
        if val:
            try:
                return datetime.fromisoformat(val)
            except ValueError:
                logging.warning(f"⚠️ Could not parse datetime from key '{key}': {val}")
        return None

    @classmethod
    def clean_orphaned_tasks_on_start(cls):
        logging.info("🔍 Checking for orphaned tasks on startup...")

        db = DBEngine("progress")
        engine = db.get_engine()

        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
                task_tables = [row[0] for row in result]
                logging.debug2(f"📋 Found {len(task_tables)} tables in 'public' schema.")

        except Exception as e:
            logging.error(f"❌ Failed to fetch task tables from database: {e}", exc_info=True)
            return

        for task_table in task_tables:
            if task_table in ["progressArchive", "profileArchive"]:
                logging.debug2(f"🚫 Skipping reserved table: {task_table}")
                continue

            logging.debug2(f"🔎 Inspecting potential orphaned task table: {task_table}")
            controller = cls(task_name=task_table, db_key="progress")

            try:
                current_status = controller.db.get("Status")
                task_uuid = controller.db.get("task_uuid")
                start_time = controller.db.get("start_time")

                logging.debug2(
                    f"📄 Task '{task_table}' - status: {current_status}, UUID: {task_uuid}, started at: {start_time}"
                )

                logging.warning(f"⚠️ Task '{task_table}' was marked {str(current_status)}. Marking as Wiped & archiving...")

                controller.db.set("Status", "Wiped")
                controller.db.set("message", "Cleaned up after crash. old status: " + str(current_status))
                controller.db.set("Finished", "False")

                controller.archive_with_orm(force_replace=True)
                logging.info(f"✅ Cleaned and archived orphaned task: {task_table}")



            except Exception as e:
                logging.error(f"❌ Failed to clean or inspect task '{task_table}': {e}", exc_info=True)





    @staticmethod
    def fetch_all_status():
        
        db = DBEngine("progress")
        engine = db.get_engine()
        results = []
        
        with engine.connect() as conn:
            res = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
            filtered_tables = [row[0] for row in res if row[0] not in {"progressArchive", "profileArchive"}]
            
            for table_name in filtered_tables:
                sql_dict = SQL_Dict(db_key="progress", table_name=table_name)
                try:
                    status = sql_dict.get("Status")
                    if status != "Ready":
                        results.append({
                            "task_name": table_name,
                            "status": status,
                            "message": sql_dict.get("message"),
                            "progress": float(sql_dict.get("progress") or 0.0)
                        })
                except Exception as e:
                    logging.warning(f"⚠️ Failed to read status for {table_name}: {e}")
        return results
    


    def progress_table_exists(self) -> bool:
        """Check if the progress table for this task still exists in the database."""
        engine = self.db.get_engine()
        try:
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT 1 FROM pg_tables
                        WHERE schemaname = 'public'
                        AND tablename = :task_table
                    )
                """), {"task_table": self.db.table_name})
                exists = result.scalar()
                logging.debug2(f"🔍 Checked existence of progress table '{self.db.table_name}': {exists}")
                return exists
        except Exception as e:
            logging.error(f"❌ Failed to check existence of progress table '{self.db.table_name}': {e}", exc_info=True)
            return False


    @staticmethod
    def is_task_done(task_name: str) -> bool:
        """
        Check whether a task has finished by verifying if its progress table exists.
        If the table is missing, assume it's completed and archived.
        """
        db = DBEngine("progress")
        engine = db.get_engine()

        query = f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = :table_name
            )
        """
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), {"table_name": task_name}).scalar()
                return not result  # If table doesn't exist, task is done
        except Exception as e:
            logging.error(f"❌ Error checking task status for '{task_name}': {e}")
            return True  # Assume done if there's an error

    def watch_task_completion(
        task_names: Union[str, List[str]],
        timeout_sec: int = 30,
        poll_interval: float = 1.0
    ) -> bool:
        """
        Polls for completion of one or more tasks by checking if their progress tables are deleted.

        :param task_names: Single task name or list of task names.
        :param timeout_sec: Maximum time to wait (per task) in seconds.
        :param poll_interval: Time to wait between checks.
        :return: True if all tasks completed within timeout, False otherwise.
        """
        if isinstance(task_names, str):
            task_names = [task_names]

        logging.info(f"🔍 Watching {len(task_names)} task(s) for completion (timeout={timeout_sec}s each)...")

        start_times = {task: time.time() for task in task_names}
        remaining_tasks = set(task_names)

        while remaining_tasks:
            for task in list(remaining_tasks):
                if TaskController.is_task_done(task):
                    logging.info(f"✅ Task '{task}' completed in {int(time.time() - start_times[task])}s.")
                    remaining_tasks.remove(task)
                elif time.time() - start_times[task] > timeout_sec:
                    logging.warning(f"⚠️ Timeout reached for task '{task}' after {timeout_sec}s.")
                    remaining_tasks.remove(task)

            if remaining_tasks:
                logging.debug2(f"Waiting for {task} for {int(time.time() - start_times[task])}s of the {timeout_sec} timeout secs.")
                time.sleep(poll_interval)

        return len(remaining_tasks) == 0