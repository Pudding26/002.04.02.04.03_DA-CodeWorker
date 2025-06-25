# Pydantic Schema + API: app/utils/SQL/models/raw/api/api_DS09Entry.py

from pydantic import BaseModel, ConfigDict
from typing import Optional, List, ClassVar, Any
from sqlalchemy.orm import Session
import logging

from app.utils.common.app.utils.SQL.models.api_BaseModel import api_BaseModel

from app.utils.common.app.utils.SQL.models.production.orm.DS09 import DS09


class DS09_Out(api_BaseModel):
    
    orm_class: ClassVar = DS09
    db_key: ClassVar[str] = "production"


    id: int
    IFAW_code: Optional[str]
    origin: Optional[str]
    engName: Optional[str]
    deName: Optional[str]
    frName: Optional[str]
    species: Optional[str]
    genus: Optional[str]

    model_config = ConfigDict(from_attributes=True)
