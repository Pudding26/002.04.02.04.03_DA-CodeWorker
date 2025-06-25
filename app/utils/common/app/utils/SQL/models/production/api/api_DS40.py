# Pydantic Schema + API: app/utils/SQL/models/raw/api/api_DS09Entry.py

from pydantic import BaseModel, ConfigDict
from typing import Optional, List, ClassVar, Any
from sqlalchemy.orm import Session
import logging
from app.utils.common.app.utils.SQL.models.api_BaseModel import api_BaseModel

from app.utils.common.app.utils.SQL.models.production.orm.DS40 import DS40

class DS40_Out(api_BaseModel):
    


    orm_class: ClassVar = DS40
    db_key: ClassVar[str] = "production"

    id: int
    genus: int
    family: str


