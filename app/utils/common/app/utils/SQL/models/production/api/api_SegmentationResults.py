from typing import Optional
from app.utils.SQL.models.api_BaseModel import api_BaseModel

class SegmentationResultBase(api_BaseModel):

    orm_class: ClassVar = SegmentationResults
    db_key: ClassVar[str] = "production"

    sampleID: str
    shotID: Optional[str]
    position: Optional[int]
    bin_type: Optional[str]
    percentile_bin: Optional[float]
    feature_name: Optional[str]
    stat_type: Optional[str]
    feature_value: Optional[float]
    unit: Optional[str]
    object_count: Optional[int]
