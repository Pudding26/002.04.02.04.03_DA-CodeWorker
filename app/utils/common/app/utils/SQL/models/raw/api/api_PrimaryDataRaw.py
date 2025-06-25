from pydantic import BaseModel, ConfigDict
from typing import Optional, List, ClassVar, Any


from app.utils.common.app.utils.SQL.models.api_BaseModel import api_BaseModel

from app.utils.common.app.utils.SQL.models.raw.orm.PrimaryDataRaw import PrimaryDataRaw

class PrimaryDataRaw_Out(api_BaseModel):

    orm_class: ClassVar = PrimaryDataRaw


    raw_UUID: str
    
    species: str

    woodType: Optional[str]
    family: Optional[str]
    genus: Optional[str]

    sourceNo: Optional[str]
    source_UUID: Optional[str]

    shotNo: Optional[int]
    specimenNo: int
    totalNumberShots: Optional[int]

    view: str
    lens: Optional[float]
    pixel_x: Optional[int]
    pixel_y: Optional[int]
    numericalAperature_NA: Optional[float]

    citeKey: Optional[str]


    #bitDepth: Optional[Any]
    #colorSpace: Optional[Any]
    #colorDepth: Optional[Any]
    

    DPI: Optional[float]
    area_x_mm: Optional[float]
    area_y_mm: Optional[float]
    pixelSize_um_per_pixel: Optional[float]



    filename: Optional[Any]
    sourceStoredLocally: Optional[Any]
    sourceFilePath_rel: Optional[Any]

    institution: Optional[Any]
    contributor: Optional[Any]
    origin: Optional[Any]
    digitizedDate: Optional[Any]
    institutionCode: Optional[Any]

    japName: Optional[Any]
    samplingPoint: Optional[Any]
    microscopicTechnic: Optional[Any]
    anatomy1_DS04: Optional[Any]
    anatomy2_DS04: Optional[Any]

    GPS_Alt: Optional[Any]
    GPS_Lat: Optional[Any]
    GPS_Long: Optional[Any]



    filename_drop: Optional[Any]
    max_split_drop: Optional[Any]
    section_drop: Optional[Any]
    subgenus_drop: Optional[Any]
    otherNo_drop: Optional[Any]
    prepNo_drop: Optional[Any]
    subspecies_drop: Optional[Any]
    individuals_drop: Optional[Any]
    n_individuals_drop: Optional[Any]
    name_drop: Optional[Any]
    source_drop: Optional[Any]
    

    version_old: Optional[str]
    specimenID_old: Optional[Any]
    specimenNo_old: Optional[Any]
    order_old: Optional[Any]
    lens_old: Optional[Any]
    engName_old: Optional[Any]
    contributor_old: Optional[Any]
    institution_old: Optional[Any]
    filename_old: Optional[Any]
    view_old: Optional[Any]
    todo_bitdepth_old: Optional[Any]
    woodType_old: Optional[Any]






