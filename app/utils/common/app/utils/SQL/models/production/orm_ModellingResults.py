from app.utils.common.app.utils.SQL.models.orm_BaseModel import orm_BaseModel

from sqlalchemy import Column, String, Integer, Float, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB


from sqlalchemy.orm import relationship


class orm_ModellingResults(orm_BaseModel):
    __tablename__ = "modellingResults"



    # Primary key
    validation_UUID = Column(String(length=255), primary_key=True)

    DoE_UUID = Column(String(length=255), index=True)
    bootstrap_no = Column(Integer, nullable=True)
    fold_no = Column(Integer, nullable=True)


    scope = Column(String, index=True)
    frac = Column(Float, index=True)
    label = Column(String, index=True)

    # Prediction metrics
    rf_acc = Column(Float, nullable=True)
    knn_acc = Column(Float, nullable=True)

    initial_col_count = Column(Integer, nullable=True)
    initial_row_count = Column(Integer, nullable=True)
    validation_col_count = Column(Integer, nullable=True)
    validation_row_count = Column(Integer, nullable=True)

    # family
    family_n_unique_train = Column(Integer, nullable=True)
    family_entropy_train = Column(Float, nullable=True)
    family_n_unique_test = Column(Integer, nullable=True)
    family_entropy_test = Column(Float, nullable=True)

    # genus
    genus_n_unique_train = Column(Integer, nullable=True)
    genus_entropy_train = Column(Float, nullable=True)
    genus_n_unique_test = Column(Integer, nullable=True)
    genus_entropy_test = Column(Float, nullable=True)

    # species
    species_n_unique_train = Column(Integer, nullable=True)
    species_entropy_train = Column(Float, nullable=True)
    species_n_unique_test = Column(Integer, nullable=True)
    species_entropy_test = Column(Float, nullable=True)

    # sourceID
    sourceID_n_unique_train = Column(Integer, nullable=True)
    sourceID_entropy_train = Column(Float, nullable=True)
    sourceID_n_unique_test = Column(Integer, nullable=True)
    sourceID_entropy_test = Column(Float, nullable=True)

    # specimenID
    specimenID_n_unique_train = Column(Integer, nullable=True)
    specimenID_entropy_train = Column(Float, nullable=True)
    specimenID_n_unique_test = Column(Integer, nullable=True)
    specimenID_entropy_test = Column(Float, nullable=True)

    # sampleID
    sampleID_n_unique_train = Column(Integer, nullable=True)
    sampleID_entropy_train = Column(Float, nullable=True)
    sampleID_n_unique_test = Column(Integer, nullable=True)
    sampleID_entropy_test = Column(Float, nullable=True)

    # stackID
    stackID_n_unique_train = Column(Integer, nullable=True)
    stackID_entropy_train = Column(Float, nullable=True)
    stackID_n_unique_test = Column(Integer, nullable=True)
    stackID_entropy_test = Column(Float, nullable=True)

    # shotID
    shotID_n_unique_train = Column(Integer, nullable=True)
    shotID_entropy_train = Column(Float, nullable=True)
    shotID_n_unique_test = Column(Integer, nullable=True)
    shotID_entropy_test = Column(Float, nullable=True)    


    # HDBSCAN clustering metrics (default + adaptive percentiles)
    ari_default = Column(Float, nullable=True)
    nmi_default = Column(Float, nullable=True)

    ari_adaptive_p05 = Column(Float, nullable=True)
    nmi_adaptive_p05 = Column(Float, nullable=True)

    ari_adaptive_p10 = Column(Float, nullable=True)
    nmi_adaptive_p10 = Column(Float, nullable=True)

    ari_adaptive_p20 = Column(Float, nullable=True)
    nmi_adaptive_p20 = Column(Float, nullable=True)

    ari_adaptive_p30 = Column(Float, nullable=True)
    nmi_adaptive_p30 = Column(Float, nullable=True)

    ari_adaptive_p40 = Column(Float, nullable=True)
    nmi_adaptive_p40 = Column(Float, nullable=True)

    ari_adaptive_p50 = Column(Float, nullable=True)
    nmi_adaptive_p50 = Column(Float, nullable=True)

    ari_adaptive_p75 = Column(Float, nullable=True)
    nmi_adaptive_p75 = Column(Float, nullable=True)



