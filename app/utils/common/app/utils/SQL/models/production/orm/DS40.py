from sqlalchemy import Column, String, Integer
from app.utils.common.app.utils.SQL.models.orm_BaseModel import orm_BaseModel

class DS40(orm_BaseModel):
    __tablename__ = "DS40"

    genus = Column(String(length=255), primary_key=True)
    family = Column(String(length=255))
