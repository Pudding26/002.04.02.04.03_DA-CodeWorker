from sqlalchemy import Column, String, Integer
from app.utils.SQL.models.orm_BaseModel import orm_BaseModel

class DS09(orm_BaseModel):
    __tablename__ = "DS09"

    id = Column(Integer, primary_key=True, autoincrement=True)
    species = Column(String(length=255))
    IFAW_code = Column(String(length=255))
    origin = Column(String(length=255))
    engName = Column(String(length=255))
    deName = Column(String(length=255))
    frName = Column(String(length=255))
    genus = Column(String(length=255))
