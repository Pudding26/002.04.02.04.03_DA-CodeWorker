from sqlalchemy import Column, String, Integer, Float, ForeignKey
from sqlalchemy.orm import relationship
from app.utils.common.app.utils.SQL.models.orm_BaseModel import orm_BaseModel

class orm_SegmentationResults(orm_BaseModel):
    __tablename__ = "segmentationResults"

    ROWID = Column(Integer, primary_key=True, autoincrement=True)

    shotID = Column(String(length=255), nullable=False, index=True)
    stackID = Column(String(length=255), nullable=False, index=True)
    bin_label = Column(String(length=50), nullable=False)
    bin_type = Column(String(length=50), nullable=False)

    bin_count = Column(Integer, nullable=False)
    bin_fraction = Column(Float, nullable=False)

    feature_name = Column(String(length=50), nullable=False)
    stat_type = Column(String(length=50), nullable=False)
    feature_value = Column(Float, nullable=False)

    unit = Column(String(length=50), nullable=True)