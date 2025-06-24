from sqlalchemy import Column, String, Integer, Float
from app.utils.SQL.models.orm_BaseModel import orm_BaseModel


class WoodTableB(orm_BaseModel):
    __tablename__ = "WoodTableB"

    # ──────────────────────────
    # required / non-optional
    # ──────────────────────────
    raw_UUID = Column(String(length=255), primary_key=True)

    woodType = Column(String(length=255))
    species  = Column(String(length=255))
    family   = Column(String(length=255))
    genus    = Column(String(length=255))

    view     = Column(String(length=255))
    lens     = Column(Float)          # matches float in Pydantic
    totalNumberShots = Column(Integer)  # ← was String(length=255), now int

    # Pixels & resolution
    DPI                     = Column(String(length=255))     # stays str (unknown in input)
    pixelSize_um_per_pixel  = Column(String(length=255))     # stays str
    pixel_x = Column(Integer)
    pixel_y = Column(Integer)

    # Microscopy & area
    microscopicTechnic      = Column(String(length=255))
    area_x_mm               = Column(Float)      # ← was String(length=255), now float
    area_y_mm               = Column(Float)      # ← was String(length=255), now float
    numericalAperature_NA   = Column(Float)      # ← was String(length=255), now float

    # Taxonomic / naming
    IFAW_code = Column(String(length=255))
    engName   = Column(String(length=255))
    deName    = Column(String(length=255))
    frName    = Column(String(length=255))
    japName   = Column(String(length=255))

    # Citation & provenance
    citeKey         = Column(String(length=255))
    institution     = Column(String(length=255))
    institutionCode = Column(String(length=255))
    contributor     = Column(String(length=255))
    digitizedDate   = Column(String(length=255))
    sourceNo        = Column(String(length=255))

    # File paths & linkage
    filename            = Column(String(length=255))
    sourceFilePath_rel  = Column(String(length=255))  # keep as String(length=255); switch to ARRAY/JSON if you need a real list
    shotNo              = Column(Integer)
    specimenNo          = Column(Integer)

    # GPS / sampling
    GPS_Alt        = Column(String(length=255))
    GPS_Lat        = Column(String(length=255))
    GPS_Long       = Column(String(length=255))
    samplingPoint  = Column(String(length=255))
    origin         = Column(String(length=255))

    # Anatomy descriptors
    anatomy1_DS04 = Column(String(length=255))
    anatomy2_DS04 = Column(String(length=255))

    # Source bookkeeping
    sourceStoredLocally = Column(Integer)  # ← was Float, now int
    source_UUID         = Column(String(length=255))
    
