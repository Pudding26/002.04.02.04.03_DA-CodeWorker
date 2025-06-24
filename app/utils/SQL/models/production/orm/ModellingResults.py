from app.utils.SQL.models.orm_BaseModel import orm_BaseModel

from sqlalchemy import Column, String, Integer, Float, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB


from sqlalchemy.orm import relationship


class ModellingResults(orm_BaseModel):
    __tablename__ = "modellingResults"


    DoE_UUID = Column(String(length=255), primary_key=True)  # Foreign key to DoEArchive

    ## DoeFactors ##
    ### PrimaryDataFactors##
    sourceNo = Column(JSONB)
    woodType = Column(JSONB)
    family = Column(JSONB)
    genus = Column(JSONB)
    species = Column(JSONB)
    
    filterNo = Column(JSONB)

    view = Column(JSONB)
    lens = Column(JSONB)
    
    maxShots = Column(JSONB)
    
    noShotsRange = Column(JSONB)
    
    ### SecondaryDataFactors##
    secondaryDataBins = Column(JSONB)

    ### 2. PreprocessingFactors##
    preProcessingNo = Column(JSONB)


    ### 3. MetricModelFactors##
    featureBins = Column(JSONB)
    metricModelNo = Column(JSONB)

    ## ModellingResults ##
