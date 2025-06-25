from typing import List, ClassVar, Any
from app.utils.common.app.utils.SQL.models.api_BaseModel import api_BaseModel
from app.utils.common.app.utils.SQL.models.jobs.orm_DoEJobs import orm_DoEJobs

from datetime import datetime

class DoEJobs_Out(api_BaseModel):
    orm_class: ClassVar = orm_DoEJobs
    db_key: ClassVar[str] = "jobs"

    job_uuid: str
    job_type: str
    
    status: str
    attempts: int
    next_retry: datetime
    provider_status: str
    segmenter_status: str
    extractor_status: str
    modeler_status: str
    validator_status: str
    
    created: datetime
    updated: datetime
    
    payload: dict
    parent_job_uuids: List[str]



