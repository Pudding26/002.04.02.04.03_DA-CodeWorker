from typing import Optional, ClassVar
from app.utils.common.app.utils.SQL.models.api_BaseModel import api_BaseModel
from app.utils.common.app.utils.SQL.models.production.orm.WoodTableB import WoodTableB

class WoodTableB_Out(api_BaseModel):
    orm_class: ClassVar = WoodTableB
    db_key: ClassVar[str] = "production"


    raw_UUID: str

    woodType: api_BaseModel.Enums.woodTypeEnum #str -> 
    species: str
    family: str
    genus: str

    view: str
    lens: float
    totalNumberShots: int




    DPI: Optional[str] #map as unknown
    pixelSize_um_per_pixel: Optional[str]
    #bitDepth: Optional[str]
    #colorSpace: Optional[str]
    #colorDepth: Optional[str]
    pixel_x: Optional[int]
    pixel_y: Optional[int]


    microscopicTechnic: Optional[str]
    area_x_mm: Optional[float]
    area_y_mm: Optional[float]
    numericalAperature_NA: Optional[float]


    IFAW_code: Optional[str]
    engName: Optional[str]
    deName: Optional[str]
    frName: Optional[str]
    japName: Optional[str]


    citeKey: Optional[str]
    institution: Optional[str]
    institutionCode: Optional[str]
    contributor: Optional[str]
    digitizedDate: Optional[str]
    sourceNo: str



    filename: Optional[str]
    #path: str
    sourceFilePath_rel: str

    shotNo: Optional[int]
    specimenNo: int

    GPS_Alt: Optional[str]
    GPS_Lat: Optional[str]
    GPS_Long: Optional[str]
    samplingPoint: Optional[str]


    origin: Optional[str]

    anatomy1_DS04: Optional[str]
    anatomy2_DS04: Optional[str]


    sourceStoredLocally: int
    source_UUID: str
