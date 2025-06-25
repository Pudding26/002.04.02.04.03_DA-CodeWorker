from sqlalchemy import Column, String, Float, Integer, DateTime
from app.utils.common.app.utils.SQL.models.orm_BaseModel import orm_BaseModel
from datetime import datetime
from typing import Optional

class ProgressArchive(orm_BaseModel):
    __tablename__ = "progressArchive"

    task_uuid: str = Column(String, primary_key=True, nullable=False)
    task_name: str = Column(String, nullable=False)
    start_time: Optional[datetime] = Column(DateTime)
    finish_time: Optional[datetime] = Column(DateTime)
    status: Optional[str] = Column(String)
    finished: Optional[str] = Column(String)
    message: Optional[str] = Column(String)
    progress: Optional[float] = Column(Float)
    elapsed_time: Optional[float] = Column(Float)
    total_size: Optional[float] = Column(Float)
    data_transferred_gb: Optional[float] = Column(Float)
    item_count: Optional[int] = Column(Integer)
    stack_count: Optional[int] = Column(Integer)
