from typing import Optional, ClassVar
from app.utils.common.app.utils.SQL.models.api_BaseModel import api_BaseModel

from app.utils.common.app.utils.SQL.models.production.orm_ClusterEmbeddings import orm_ClusterEmbeddings

class ClusterEmbeddings_Out(api_BaseModel):

    orm_class: ClassVar = orm_ClusterEmbeddings
    db_key: ClassVar[str] = "production"

    id: Optional[int] = None


    DoE_UUID: str
    frac: float
    scope: str

    # Hierarchical labels
    family: Optional[str]
    genus: Optional[str]
    species: Optional[str]
    sourceID: Optional[str]
    specimenID: Optional[str]
    sampleID: Optional[str]
    stackID: Optional[str]
    shotID: Optional[str]


    x: float
    y: float
 
