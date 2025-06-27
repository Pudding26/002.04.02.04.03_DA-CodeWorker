from __future__ import annotations
from typing import List, Optional, Union, Dict, Any
from uuid import uuid4
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field

import base64
import numpy as np
from pydantic import BaseModel, field_validator

#ORM
from app.utils.common.app.utils.SQL.models.jobs.orm_JobLink import orm_JobLink



from app.utils.common.app.utils.dataModels.Jobs.BaseJob import BaseJob
from app.utils.common.app.utils.dataModels.Jobs.DoEJob import DoEJob
from app.utils.common.app.utils.dataModels.Jobs.ExtractorJob import ExtractorJobInput


from app.utils.common.app.utils.dataModels.Jobs.JobEnums import JobStatus, JobKind, RelationState


from app.utils.common.app.utils.SQL.models.jobs.orm_WorkerJobs import orm_WorkerJobs

from uuid import UUID

from sqlalchemy import event, delete, insert, update, select, func
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Mapper




class ModelerJob(BaseJob):
    
    
    job_type: JobKind = JobKind.MODELER
    orm_model = orm_WorkerJobs
    status: JobStatus = JobStatus.READY.value
    input: ModelerJobInput
    attrs: ModelerAttrs
    stats: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class ModelerJobInput(BaseModel):

    stackIDs: List[str]

    metricModel_instructions: Dict[str, Any] = Field(default_factory=dict)
    preprocessing_instructions: Dict[str, Any] = Field(default_factory=dict)
    
    job_No: Optional[int] = None

    preProcessingNo: str
    metricModelNo: str


    model_config = ConfigDict(arbitrary_types_allowed=True)

class ModelerAttrs(BaseModel):
    raw_data: Optional[Any] = None  # Pandas
    preprocessed_data: Optional[Any] # cupy
    model_results: Optional[Any] #pandas


