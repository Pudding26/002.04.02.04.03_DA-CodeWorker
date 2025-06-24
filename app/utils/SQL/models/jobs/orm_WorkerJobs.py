from app.utils.SQL.models.orm_BaseModel import orm_BaseModel
from app.utils.SQL.DBEngine import DBEngine


from sqlalchemy.orm import relationship
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy import Column, String, Integer
from sqlalchemy.dialects.postgresql import JSONB

from sqlalchemy import DateTime


from sqlalchemy import update
from contextlib import contextmanager

class orm_WorkerJobs(orm_BaseModel):
    __tablename__ = "WorkerJobs"

    job_uuid = Column(String, primary_key=True)
    job_type = Column(String)
    
    status = Column(String)
    attempts = Column(Integer)
    next_retry = Column(DateTime)
    
    created = Column(DateTime)
    updated = Column(DateTime)
    
    payload = Column(JSONB)
    parent_job_uuids = Column(JSONB)

    parent_links = relationship(
        "orm_JobLink",
        back_populates="child_worker",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    linked_doe_jobs = association_proxy(
        "parent_links",
        "parent_doe"
    )



    @classmethod
    def update_row(cls, row: dict):
        job_uuid = row.pop("job_uuid", None)
        if not job_uuid:
            raise ValueError("Missing job_uuid in row")
        session = DBEngine("jobs").get_session()

        obj = session.get(cls, job_uuid)
        if not obj:
            raise LookupError(f"WorkerJob {job_uuid} not found")

        for key, value in row.items():
            setattr(obj, key, value)

        session.commit()    # <-- Commit the changes!
        session.close()     # <-- Clean up the session

        return obj




