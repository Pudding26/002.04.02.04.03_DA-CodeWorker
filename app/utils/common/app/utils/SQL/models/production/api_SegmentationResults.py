from typing import Optional, ClassVar
from app.utils.common.app.utils.SQL.models.api_BaseModel import api_BaseModel

from app.utils.common.app.utils.SQL.models.production.orm_SegmentationResults import orm_SegmentationResults

class SegmentationResults_Out(api_BaseModel):

    orm_class: ClassVar = orm_SegmentationResults
    db_key: ClassVar[str] = "production"

    ROWID: Optional[int] = None

    shotID: str
    stackID: str
    bin_label: str
    bin_type: str

    bin_count: int
    bin_fraction: float

    feature_name: str
    stat_type: str
    feature_value: float

    unit: Optional[str]
