from sqlalchemy import Column, String, Float, Integer, DateTime
from sqlalchemy.dialects.postgresql import UUID
from app.utils.common.app.utils.SQL.models.orm_BaseModel import orm_BaseModel
import uuid
from datetime import datetime


class ProfileArchive(orm_BaseModel):
    __tablename__ = "profileArchive"

    id = Column(Integer, primary_key=True, index=True)
    task_uuid = Column(UUID(as_uuid=True), nullable=False, index=True)
    task_name = Column(String, nullable=False)
    task_group = Column(String, nullable=True)
    device = Column(String, nullable=True)
    profile_type = Column(String, nullable=True)
    time = Column(DateTime, default=datetime.now)

    line_number = Column(Integer)
    mem_usage = Column(Float)
    increment = Column(Float)
    occurrences = Column(Integer)
    line_contents = Column(String)
