# ORM Model: app/utils/SQL/models/raw/orm/PrimaryDataRaw.py

from sqlalchemy import Column, String, Integer, Boolean, BigInteger, Float
from app.utils.SQL.models.orm_BaseModel import orm_BaseModel





class PrimaryDataRaw(orm_BaseModel):
    __tablename__ = "primaryDataRaw"
    
    
    raw_UUID = Column(String, primary_key=True)

    species = Column(String)

    woodType = Column(String)
    family = Column(String)
    genus = Column(String)

    sourceNo = Column(String)
    source_UUID = Column(String)

    shotNo = Column(Integer)
    specimenNo = Column(Integer)
    totalNumberShots = Column(Integer)

    view = Column(String)
    lens = Column(Float)
    pixel_x = Column(Integer)
    pixel_y = Column(Integer)
    numericalAperature_NA = Column(Float)

    citeKey = Column(String)

    #bitDepth = Column(String)
    #colorSpace = Column(String)
    #colorDepth = Column(String)

    DPI = Column(Float)
    area_x_mm = Column(Float)
    area_y_mm = Column(Float)
    pixelSize_um_per_pixel = Column(Float)





    filename = Column(String)
    sourceFilePath_rel = Column(String)
    sourceStoredLocally = Column(String)

    institution = Column(String)
    contributor = Column(String)
    origin = Column(String)
    digitizedDate = Column(String)
    institutionCode = Column(String)

    GPS_Alt = Column(String)
    GPS_Lat = Column(String)
    GPS_Long = Column(String)

    filename_drop = Column(String)
    max_split_drop = Column(String)
    section_drop = Column(String)
    subgenus_drop = Column(String)
    otherNo_drop = Column(String)
    prepNo_drop = Column(String)
    subspecies_drop = Column(String)
    individuals_drop = Column(String)
    n_individuals_drop = Column(String)



    version_old = Column(String)
    specimenNo_old = Column(String)
    specimenID_old = Column(String)
    order_old = Column(String)
    lens_old = Column(String)
    woodType_old = Column(String)

    japName = Column(String)
    samplingPoint = Column(String)
    microscopicTechnic = Column(String)
    name_drop = Column(String)
    filename_old = Column(String)
    source_drop = Column(String)
    todo_bitdepth_old = Column(String)
    engName_old = Column(String)
    contributor_old = Column(String)
    institution_old = Column(String)
    view_old = Column(String)
    anatomy1_DS04 = Column(String)
    anatomy2_DS04 = Column(String)
