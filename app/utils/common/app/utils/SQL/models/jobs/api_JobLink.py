from typing import List, ClassVar, Any
from app.utils.common.app.utils.SQL.models.api_BaseModel import api_BaseModel
from app.utils.common.app.utils.SQL.models.temp.orm.JobLink import JobLink

from pydantic import Field

from app.utils.common.app.utils.dataModels.Jobs.JobEnums import JobKind, RelationState



from datetime import datetime

class JobLink_Out(api_BaseModel):
    orm_class: ClassVar = JobLink
    db_key: ClassVar[str] = "jobs"


    parent_uuid: str = Field(..., description="UUID of the parent DoEJob")
    child_uuid: str = Field(..., description="UUID of the child job")
    child_kind: JobKind = Field(..., description="Kind of the child job")
    rel_state: RelationState = Field(..., description="Current relationship state")


