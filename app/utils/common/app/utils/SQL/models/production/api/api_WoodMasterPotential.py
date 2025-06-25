from typing import Optional, List, ClassVar, Any
from datetime import datetime
from app.utils.common.app.utils.SQL.models.api_BaseModel import api_BaseModel


from app.utils.common.app.utils.SQL.models.production.orm.WoodMasterPotential import WoodMasterPotential  # Assuming this ORM class exists

from app.utils.common.app.utils.SQL.models.enums import sampleID_statusEnum


class WoodMasterPotential_Out(api_BaseModel):


    orm_class: ClassVar = WoodMasterPotential
    db_key: ClassVar[str] = "production"

    sampleID: str
    #sampleID_status: sampleID_statusEnum
    transfer_trys: int = 0


    woodType: str
    species: str
    family: str
    genus: str

    view: str

    lens: float
    totalNumberShots: int

    #filterNo: str
    DPI: Optional[float] = None
    pixelSize_um_per_pixel: Optional[float] = None
    #bitDepth: float
    #colorDepth: str
    #colorSpace: str
    #pixel_x: float
    #pixel_y: float
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
    raw_UUID: Any  # JSONB type in SQLAlchemy, changed to Any for flexibility
    source_UUID: Any  # JSONB type in SQLAlchemy, using Any for flexibility

    GPS_Alt: Optional[float]  = None
    GPS_Lat: Optional[float]  = None
    GPS_Long: Optional[float]  = None

    hdf5_dataset_path: str # The path to the later dataset. excluding the stackID
    sourceFilePath_rel: Any # JSONB type in SQLAlchemy, using Any for flexibility
    specimenID: str
    sourceID: str
    sourceStoredLocally : Any  # JSONB type in SQLAlchemy, using Any for flexibility
