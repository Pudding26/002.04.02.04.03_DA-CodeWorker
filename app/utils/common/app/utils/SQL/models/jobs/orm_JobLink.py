from enum import Enum
from sqlalchemy.dialects.postgresql import UUID

from sqlalchemy import (
    Column,
    String,
    Enum as PgEnum,
    ForeignKey,
    Table,
    UUID as SQLUUID,
    create_engine,
)
from sqlalchemy.orm import relationship

from app.utils.common.app.utils.dataModels.Jobs.JobEnums import JobKind, RelationState

from app.utils.common.app.utils.SQL.models.orm_BaseModel import orm_BaseModel


class orm_JobLink(orm_BaseModel):
    """
    Association-object pattern.
    We don’t declare a FK on child_uuid because it could live in any job table.
    """
    __tablename__ = "jobLink"

    print("✅ orm_JobLink loaded at", __name__)

    parent_uuid = Column(String, ForeignKey("DoEJobs.job_uuid", ondelete="CASCADE"), primary_key=True)
    child_uuid = Column(String, ForeignKey("WorkerJobs.job_uuid", ondelete="CASCADE"), primary_key=True)

    child_kind = Column(String, nullable=False)
    rel_state = Column(String, nullable=False, default=RelationState.IN_PROGRESS)

    parent_doe = relationship("orm_DoEJobs", back_populates="child_links")
    child_worker = relationship("orm_WorkerJobs", back_populates="parent_links")


    
    
    


    

    
