from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from app.utils.common.app.utils.SQL.models.orm_BaseModel import orm_BaseModel

class WoodMaster(orm_BaseModel):
    __tablename__ = "woodMaster"

    stackID = Column(String(length=255), primary_key=True)
    sampleID = Column(String(length=255))

    woodType = Column(String(length=255))
    species = Column(String(length=255))
    family = Column(String(length=255))
    genus = Column(String(length=255))

    view = Column(String(length=255))
    lens = Column(Float)
    totalNumberShots = Column(Integer)

    filterNo = Column(String(length=255))  # changed from Integer
    DPI = Column(Float)        # was Integer
    pixelSize_um_per_pixel = Column(Float)
    bitDepth = Column(Float)   # was Integer
    colorDepth = Column(String(length=255))
    colorSpace = Column(String(length=255))
    pixel_x = Column(Float)    # was Integer
    pixel_y = Column(Float)    # was Integer

    microscopicTechnic = Column(String(length=255))
    area_x_mm = Column(Float)
    area_y_mm = Column(Float)
    numericalAperature_NA = Column(Float)

    IFAW_code = Column(String(length=255))
    engName = Column(String(length=255))
    deName = Column(String(length=255))
    frName = Column(String(length=255))
    japName = Column(String(length=255))
    origin = Column(String(length=255))

    citeKey = Column(String(length=255))
    institution = Column(String(length=255))
    institutionCode = Column(String(length=255))
    contributor = Column(String(length=255))
    digitizedDate = Column(DateTime)
    sourceNo = Column(String(length=255))
    raw_UUID = Column(String(length=255), nullable=False)

    GPS_Alt = Column(Float)
    GPS_Lat = Column(Float)
    GPS_Long = Column(Float)
    samplingPoint = Column(String(length=255))


    path = Column(String(length=255))  # ✅ newly added
    specimenID = Column(String(length=255))
    sourceID = Column(String(length=255))
    was_cropped = Column(Boolean)
