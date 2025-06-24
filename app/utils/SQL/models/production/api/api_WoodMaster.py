from typing import Optional, List, ClassVar
from datetime import datetime
from app.utils.SQL.models.api_BaseModel import api_BaseModel


from app.utils.SQL.models.production.orm.WoodMaster import WoodMaster  # Assuming this ORM class exists

class WoodMaster_Out(api_BaseModel):


    orm_class: ClassVar = WoodMaster
    db_key: ClassVar[str] = "production"

    stackID: str
    sampleID: str

    woodType: str
    species: str
    family: str
    genus: str

    view: str

    lens: float
    totalNumberShots: int

    filterNo: str
    DPI: Optional[float] = None
    pixelSize_um_per_pixel: Optional[float] = None
    bitDepth: float
    colorDepth: str
    colorSpace: str
    pixel_x: float
    pixel_y: float
    microscopicTechnic: Optional[str] = None
    area_x_mm: Optional[float] = None
    area_y_mm: Optional[float] = None
    numericalAperature_NA: Optional[float] = None

    IFAW_code: Optional[str] = None
    engName: Optional[str] = None
    deName: Optional[str] = None
    frName: Optional[str]  = None
    japName: Optional[str]  = None
    samplingPoint: Optional[str]  = None
    origin: Optional[str]  = None

    citeKey: str
    institution: Optional[str] = "unknown"
    institutionCode: Optional[str] = "unknown"
    contributor: Optional[str] = "unknown"
    digitizedDate: Optional[datetime] = None
    sourceNo: str
    raw_UUID: str

    GPS_Alt: Optional[float]  = None
    GPS_Lat: Optional[float]  = None
    GPS_Long: Optional[float]  = None

    path: str
    stackID: str
    specimenID: str
    sourceID: str
    was_cropped: bool
