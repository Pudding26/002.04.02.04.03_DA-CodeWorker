from app.utils.common.app.utils.SQL.models.orm_BaseModel import orm_BaseModel

from sqlalchemy import Column, String, Integer, Float, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB


from sqlalchemy.orm import relationship


class orm_ModellingResults(orm_BaseModel):
    __tablename__ = "modellingResults"



    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    DoE_UUID = Column(String(length=255), index=True)
    scope = Column(String, index=True)
    frac = Column(Float, index=True)
    label = Column(String, index=True)

    randomforest_acc = Column(Float, nullable=True)
    logreg_acc = Column(Float, nullable=True)
    knn_acc = Column(Float, nullable=True)

    hdbscan_ari_default = Column(Float, nullable=True)
    hdbscan_ari_adaptive = Column(Float, nullable=True)
    hdbscan_nmi_default = Column(Float, nullable=True)
    hdbscan_nmi_adaptive = Column(Float, nullable=True)
    hdbscan_silhouette_default = Column(Float, nullable=True)
    hdbscan_silhouette_adaptive = Column(Float, nullable=True)

    family_n_unique = Column(Integer, nullable=True)
    genus_n_unique = Column(Integer, nullable=True)
    species_n_unique = Column(Integer, nullable=True)
    sourceID_n_unique = Column(Integer, nullable=True)
    specimenID_n_unique = Column(Integer, nullable=True)
    sampleID_n_unique = Column(Integer, nullable=True)
    stackID_n_unique = Column(Integer, nullable=True)
    shotID_n_unique = Column(Integer, nullable=True)

    n_rows = Column(Integer, nullable=True)


