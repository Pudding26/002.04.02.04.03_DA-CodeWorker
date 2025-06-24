from __future__ import annotations
from typing import List, Optional, Union, Dict, Any
from uuid import uuid4
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field

#ORM
from app.utils.SQL.models.jobs.orm_JobLink import orm_JobLink



from app.utils.dataModels.Jobs.BaseJob import BaseJob
from app.utils.dataModels.Jobs.DoEJob import DoEJob

from app.utils.dataModels.Jobs.JobEnums import JobStatus, JobKind, RelationState

from app.utils.SQL.models.jobs.orm_WorkerJobs import orm_WorkerJobs

from uuid import UUID

from sqlalchemy import event, delete, insert, update, select, func
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Mapper




class ProviderJob(BaseJob):
    
    
    job_type: JobKind = JobKind.PROVIDER
    orm_model = orm_WorkerJobs
    status: JobStatus = JobStatus.READY.value
    input: ProviderJobInput
    attrs: ProviderAttrs
    

    model_config = ConfigDict(extra="forbid")


class ProviderJobInput(BaseModel):
    src_file_path: str
    src_ds_rel_path: Union[str, List[str]]
    dest_rel_path: str
    stored_locally: List[int]
    image_data: Optional[Any] = None
    job_No: Optional[int] = None


    model_config = ConfigDict(arbitrary_types_allowed=True)


class ProviderAttrs(BaseModel):
    Level1: Dict[str, Union[str, int]]
    Level2: Dict[str, Union[str, int]]
    Level3: Dict[str, Union[str, int]]
    Level4: Dict[str, Optional[str]]
    Level5: Dict[str, Union[str, int]]
    Level6: Dict[str, Optional[str]]
    Level7: Dict[str, Union[str, int, float, Optional[str]]]
    dataSet_attrs: Dict[str, Optional[Union[str, float, int]]]

    model_config = ConfigDict(arbitrary_types_allowed=True)
