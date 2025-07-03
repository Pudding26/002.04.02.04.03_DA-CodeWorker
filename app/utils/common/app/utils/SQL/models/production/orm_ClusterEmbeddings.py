from app.utils.common.app.utils.SQL.models.orm_BaseModel import orm_BaseModel

from sqlalchemy import Column, String, Integer, Float, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB


from sqlalchemy.orm import relationship


class orm_ClusterEmbeddings(orm_BaseModel):
    __tablename__ = "clusterEmbeddings"


    id = Column(Integer, primary_key=True, autoincrement=True)

    DoE_UUID = Column(String, index=True)
    frac = Column(Float)
    scope = Column(String, index=True)

    # Hierarchical labels
    shotID = Column(String, index=True)
    family = Column(String, index=True)
    genus = Column(String, index=True)
    species = Column(String, index=True)
    sourceID = Column(String, index=True)
    specimenID = Column(String, index=True)
    sampleID = Column(String, index=True)
    stackID = Column(String, index=True)
    shotID = Column(String, index=True)

    x = Column(Float)
    y = Column(Float)



