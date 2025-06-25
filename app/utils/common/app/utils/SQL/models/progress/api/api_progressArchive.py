from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from sqlalchemy.orm import Session
import logging

from app.utils.common.app.utils.SQL.DBEngine import DBEngine
from app.utils.common.app.utils.SQL.models.progress.orm.ProgressArchive import ProgressArchive
from app.utils.common.app.utils.SQL.models.api_BaseModel import api_BaseModel


class ProgressArchiveOut(api_BaseModel):
    task_uuid: str
    task_name: str
    start_time: Optional[datetime]
    finish_time: Optional[datetime]
    status: Optional[str]
    finished: Optional[str]
    message: Optional[str]
    progress: Optional[float]
    elapsed_time: Optional[float]
    total_size: Optional[float]
    data_transferred_gb: Optional[float]
    item_count: Optional[int]
    stack_count: Optional[int]

    @classmethod
    def persist_to_db(cls, data: "ProgressArchiveOut", force_replace: bool = False):
        """Stores the archive entry in the database, optionally replacing by UUID."""
        engine = DBEngine("progress")
        session: Session = engine.get_session()

        logging.debug2(f"📝 Persisting ProgressArchive for task '{data.task_name}' (UUID: {data.task_uuid})")
        if force_replace:
            logging.debug2("🔁 force_replace enabled — checking for existing entries")

        try:
            if force_replace:
                deleted = session.query(ProgressArchive).filter_by(task_uuid=data.task_uuid).delete()
                if deleted:
                    logging.debug2(f"🗑️ Deleted existing archive entry for UUID: {data.task_uuid}")
                else:
                    logging.debug2(f"ℹ️ No existing entry found for UUID: {data.task_uuid}")

            entry = ProgressArchive(**data.dict())
            session.add(entry)
            session.commit()

            logging.info(f"✅ Archived progress for task '{data.task_name}' (UUID: {data.task_uuid})")
            return entry

        except Exception as e:
            session.rollback()
            logging.error(f"❌ Failed to persist archive for task '{data.task_name}' (UUID: {data.task_uuid}): {e}", exc_info=True)
            raise RuntimeError(f"Failed to persist ProgressArchive: {e}")

        finally:
            session.close()
            logging.debug2("🔚 Session closed after archive persistence")

    @classmethod
    def delete_existing_by_uuid(cls, task_uuid: str):
        """Deletes existing archive entry for the same task_uuid if it exists."""
        engine = DBEngine("progress")
        session: Session = engine.get_session()

        logging.debug2(f"🔍 Checking for archive to delete with UUID: {task_uuid}")

        try:
            deleted = session.query(ProgressArchive).filter_by(task_uuid=task_uuid).delete()
            if deleted:
                logging.info(f"🗑️ Deleted existing archive entry with UUID: {task_uuid}")
            else:
                logging.debug2(f"ℹ️ No archive entry found for UUID: {task_uuid}")
            session.commit()
        except Exception as e:
            session.rollback()
            logging.error(f"❌ Failed to delete archive for UUID {task_uuid}: {e}", exc_info=True)
        finally:
            session.close()
            logging.debug2("🔚 Session closed after delete operation")
