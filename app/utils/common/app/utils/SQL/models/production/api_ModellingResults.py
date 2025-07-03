from typing import List, ClassVar, Any, Optional
from app.utils.common.app.utils.SQL.models.api_BaseModel import api_BaseModel
from app.utils.common.app.utils.SQL.models.production.orm_ModellingResults import orm_ModellingResults

class ModellingResults_Out(api_BaseModel):
    orm_class: ClassVar = orm_ModellingResults
    db_key: ClassVar[str] = "production"


    id: Optional[int] = None
    
    DoE_UUID: str
    scope: str
    frac: float
    label: str

    randomforest_acc: Optional[float] = None
    logreg_acc: Optional[float] = None
    knn_acc: Optional[float] = None

    hdbscan_ari_default: Optional[float] = None
    hdbscan_ari_adaptive: Optional[float] = None
    hdbscan_nmi_default: Optional[float] = None
    hdbscan_nmi_adaptive: Optional[float] = None
    hdbscan_silhouette_default: Optional[float] = None
    hdbscan_silhouette_adaptive: Optional[float] = None

    family_n_unique: Optional[int] = None
    genus_n_unique: Optional[int] = None
    species_n_unique: Optional[int] = None
    sourceID_n_unique: Optional[int] = None
    specimenID_n_unique: Optional[int] = None
    sampleID_n_unique: Optional[int] = None
    stackID_n_unique: Optional[int] = None
    shotID_n_unique: Optional[int] = None

    n_rows: Optional[int] = None
    


