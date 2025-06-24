from typing import List, ClassVar, Any
from app.utils.SQL.models.api_BaseModel import api_BaseModel
from app.utils.SQL.models.production.orm.ModellingResults import ModellingResults

class ModellingResults_Out(api_BaseModel):
    orm_class: ClassVar = ModellingResults
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
    
    maxShots: List[Any]
    
    noShotsRange: List[Any]
    
    ## SecondaryDataFactors##
    secondaryDataBins: List[str]


    ## 2. PreprocessingFactors##
    preProcessingNo: List[str]

    ## 3. MetricModelFactors##
    featureBins: List[str]
    metricModelNo: List[str]



