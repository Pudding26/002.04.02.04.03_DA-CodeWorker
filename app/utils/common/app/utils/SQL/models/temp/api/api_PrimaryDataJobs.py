from typing import Optional, List, ClassVar, Any
from datetime import datetime
from app.utils.SQL.models.api_BaseModel import api_BaseModel


from app.utils.SQL.models.temp.orm.PrimaryDataJobs import PrimaryDataJobs

class PrimaryDataJobs_Out(api_BaseModel):


    orm_class: ClassVar = PrimaryDataJobs
    db_key: ClassVar[str] = "temp"

    sampleID: str

    woodType: str
    species: str
    family: str
    genus: str

    view: str

    lens: float
    totalNumberShots: int

    #filterNo: Optional[int]
    DPI: Optional[float]
    pixelSize_um_per_pixel: Optional[float]
    #bitDepth: Optional[int]
    #colorDepth: Optional[str]
    #colorSpace: Optional[str]
    #pixel_x: Optional[int]
    #pixel_y: Optional[int]
    microscopicTechnic: Optional[str]
    area_x_mm: Optional[float]
    area_y_mm: Optional[float]
    numericalAperature_NA: Optional[float]

    IFAW_code: Optional[str]
    engName: Optional[str]
    deName: Optional[str]
    frName: Optional[str]
    japName: Optional[str]
    samplingPoint: Optional[str]
    origin: Optional[str]

    citeKey: Optional[str]
    institution: Optional[str]
    institutionCode: Optional[str]
    contributor: Optional[str]
    digitizedDate: Optional[str] = None
    sourceNo: str
    raw_UUID: Any  # JSONB type in SQLAlchemy, changed to Any for flexibility
    source_UUID: Any  # JSONB type in SQLAlchemy, using Any for flexibility

    GPS_Alt: Optional[float] = None
    GPS_Lat: Optional[float] = None
    GPS_Long: Optional[float] = None

    sourceFilePath_rel: Any # JSONB type in SQLAlchemy, using Any for flexibility
    hdf5_dataset_path: Optional[str]
    #stackID: Optional[str] -> not possible on this level
    specimenID: str
    sourceID: str
    sourceStoredLocally: Any


