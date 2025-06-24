from app.utils.SQL.models.orm_BaseModel import orm_BaseModel



from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import JSONB

class DoEArchive(orm_BaseModel):
    __tablename__ = "DoEArchive"

    DoE_UUID = Column(String(length=255), primary_key=True)    
    
    ## PrimaryDataFactors##
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
    
    ## SecondaryDataFactors##
    secondaryDataBins = Column(JSONB)

    ## 2. PreprocessingFactors##
    preProcessingNo = Column(JSONB)

    ## 3. SegmentationFactors##

    ## 4. ModelFactors##
    featureBins = Column(JSONB)
    metricModelNo = Column(JSONB)