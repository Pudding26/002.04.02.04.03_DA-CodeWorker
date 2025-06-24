from typing import Optional, ClassVar, List, Any
from app.utils.SQL.models.api_BaseModel import api_BaseModel
from app.utils.SQL.models.production.orm.DoEArchive import DoEArchive

class DoEArchive_Out(api_BaseModel):
    orm_class: ClassVar = DoEArchive
    db_key: ClassVar[str] = "production"


    DoE_UUID: str

    ## PrimaryDataFactors##

    sourceNo: List[str]
    woodType: List[str]
    family: List[str]
    genus: List[str]
    species: List[str]
    
    filterNo: List[str]

    view: List[str]
    lens: List[Any]
    
    maxShots: List[int]
    
    noShotsRange: List[int]
    
    ## SecondaryDataFactors##
    secondaryDataBins: List[str]


    ## 2. PreprocessingFactors##
    preProcessingNo: List[str]

    ## 3. MetricModelFactors##
    featureBins: List[str]
    metricModelNo: List[str]

