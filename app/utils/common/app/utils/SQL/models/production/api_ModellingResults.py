from typing import List, ClassVar, Any, Optional
from app.utils.common.app.utils.SQL.models.api_BaseModel import api_BaseModel
from app.utils.common.app.utils.SQL.models.production.orm_ModellingResults import orm_ModellingResults

class ModellingResults_Out(api_BaseModel):
    orm_class: ClassVar = orm_ModellingResults
    db_key: ClassVar[str] = "production"


class ModellingResults_Out(api_BaseModel):
    orm_class: ClassVar = orm_ModellingResults
    db_key: ClassVar[str] = "production"
 
    validation_UUID: str # Primary key | Hashed: DoE_UUID + scope + frac + label

    DoE_UUID: str
    bootstrap_no: int
    fold_no: int

    scope: str
    frac: float
    label: str

    # Prediction metrics
    rf_acc: Optional[float] = None
    knn_acc: Optional[float] = None

    initial_col_count: Optional[int] = None
    initial_row_count: Optional[int] = None

    validation_col_count: Optional[int] = None
    validation_row_count: Optional[int] = None


    # family
    family_n_unique_train: Optional[int] = 0
    family_entropy_train: Optional[float] = None
    family_n_unique_test: Optional[int] = 0
    family_entropy_test: Optional[float] = None

    # genus
    genus_n_unique_train: Optional[int] = 0
    genus_entropy_train: Optional[float] = None
    genus_n_unique_test: Optional[int] = 0
    genus_entropy_test: Optional[float] = None

    # species
    species_n_unique_train: Optional[int] = 0
    species_entropy_train: Optional[float] = None
    species_n_unique_test: Optional[int] = 0
    species_entropy_test: Optional[float] = None

    # sourceID
    sourceID_n_unique_train: Optional[int] = 0
    sourceID_entropy_train: Optional[float] = None
    sourceID_n_unique_test: Optional[int] = 0
    sourceID_entropy_test: Optional[float] = None

    # specimenID
    specimenID_n_unique_train: Optional[int] = 0
    specimenID_entropy_train: Optional[float] = None
    specimenID_n_unique_test: Optional[int] = 0
    specimenID_entropy_test: Optional[float] = None

    # sampleID
    sampleID_n_unique_train: Optional[int] = 0
    sampleID_entropy_train: Optional[float] = None
    sampleID_n_unique_test: Optional[int] = 0
    sampleID_entropy_test: Optional[float] = None

    # stackID
    stackID_n_unique_train: Optional[int] = 0
    stackID_entropy_train: Optional[float] = None
    stackID_n_unique_test: Optional[int] = 0
    stackID_entropy_test: Optional[float] = None

    # shotID
    shotID_n_unique_train: Optional[int] = 0
    shotID_entropy_train: Optional[float] = None
    shotID_n_unique_test: Optional[int] = 0
    shotID_entropy_test: Optional[float] = None

    # HDBSCAN clustering metrics
    ari_default: Optional[float] = None
    nmi_default: Optional[float] = None

    ari_adaptive_p05: Optional[float] = None
    nmi_adaptive_p05: Optional[float] = None

    ari_adaptive_p10: Optional[float] = None
    nmi_adaptive_p10: Optional[float] = None

    ari_adaptive_p20: Optional[float] = None
    nmi_adaptive_p20: Optional[float] = None

    ari_adaptive_p30: Optional[float] = None
    nmi_adaptive_p30: Optional[float] = None

    ari_adaptive_p40: Optional[float] = None
    nmi_adaptive_p40: Optional[float] = None

    ari_adaptive_p50: Optional[float] = None
    nmi_adaptive_p50: Optional[float] = None

    ari_adaptive_p75: Optional[float] = None
    nmi_adaptive_p75: Optional[float] = None
    


