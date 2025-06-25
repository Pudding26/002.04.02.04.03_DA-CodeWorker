from typing import List, ClassVar, Any
from app.utils.common.app.utils.SQL.models.api_BaseModel import api_BaseModel
from app.utils.common.app.utils.SQL.models.jobs.orm_WorkerJobs import orm_WorkerJobs



from datetime import datetime

class WorkerJobs_Out(api_BaseModel):
    orm_class: ClassVar = orm_WorkerJobs
    db_key: ClassVar[str] = "jobs"


    job_uuid: str
    job_type: str

    status: str
    attempts: int
    next_retry: datetime

    created: datetime
    updated: datetime
    
    payload: Any # is a JSNOB object
    parent_job_uuids: list


 