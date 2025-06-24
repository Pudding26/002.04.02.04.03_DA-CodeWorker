from app.utils.SQL.models.orm_BaseModel import orm_BaseModel


from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm import relationship
from sqlalchemy import Column, String, Integer
from sqlalchemy.dialects.postgresql import JSONB

from sqlalchemy import DateTime

class orm_DoEJobs(orm_BaseModel):
    __tablename__ = "DoEJobs"
    __table_args__ = {"extend_existing": True}

    job_uuid = Column(String, primary_key=True)
    job_type = Column(String)
    
    status = Column(String)
    
    provider_status = Column(String)
    segmenter_status = Column(String)
    extractor_status = Column(String)
    modeler_status = Column(String)
    validator_status = Column(String)

    attempts = Column(Integer)
    next_retry = Column(DateTime)
    
    created = Column(DateTime)
    updated = Column(DateTime)
    
    payload = Column(JSONB)
    parent_job_uuids = Column(JSONB)





    

    child_links = relationship(
        "orm_JobLink",
        back_populates="parent_doe",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    linked_worker_jobs = association_proxy(
        "child_links",
        "child_worker"
    )
