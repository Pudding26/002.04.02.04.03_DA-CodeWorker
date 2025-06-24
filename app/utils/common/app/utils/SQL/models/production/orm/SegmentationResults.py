from sqlalchemy import Column, String, Integer, Float, ForeignKey
from sqlalchemy.orm import relationship
from app.utils.SQL.models.orm_BaseModel import orm_BaseModel

class SegmentationResultsAPI(orm_BaseModel):
    __tablename__ = "segmentationResults"

    ROWID = Column(Integer, primary_key=True, autoincrement=True)

    sampleID = Column(String(length=255), ForeignKey("woodMaster.sampleID"), nullable=False)
    shotID = Column(String(length=255))
    position = Column(Integer)

    bin_type = Column(String(length=255))
    percentile_bin = Column(Float)

    feature_name = Column(String(length=255))
    stat_type = Column(String(length=255))
    feature_value = Column(Float)

    unit = Column(String(length=255))
    object_count = Column(Integer)

    # ✅ Add reverse relationship
    wood_sample = relationship("WoodMaster", back_populates="segmentation_results")
