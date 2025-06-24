from typing import List, ClassVar, Any
from app.utils.SQL.models.api_BaseModel import api_BaseModel
from app.utils.SQL.models.temp.orm.DoEJobs import DoEJobs

class DoEJobs_Out(api_BaseModel):
    orm_class: ClassVar = DoEJobs
    db_key: ClassVar[str] = "temp"


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



