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

from app.utils.common.app.utils.dataModels.Jobs.JobEnums import JobStatus, JobKind, RelationState

from app.utils.common.app.utils.SQL.models.jobs.orm_WorkerJobs import orm_WorkerJobs

from uuid import UUID

from sqlalchemy import event, delete, insert, update, select, func
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Mapper




class ExtractorJob(BaseJob):
    
    
    job_type: JobKind = JobKind.EXTRACTOR
    orm_model = orm_WorkerJobs
    status: JobStatus = JobStatus.READY.value
    input: ExtractorJobInput
    attrs: Dict[str, Any] = Field(default_factory=dict)
    

    model_config = ConfigDict(extra="forbid")


class ExtractorJobInput(BaseModel):

    mask: Optional[Any] = None  # Placeholder for mask serialzed mask data
    n_images: int
    width: int
    height: int
    stackID: str

    @field_validator('mask', mode='before')
    @classmethod
    def encode_mask(cls, v):
        # Allow NumPy arrays as input and auto-encode to base64 string
        if isinstance(v, np.ndarray):
            return base64.b64encode(v.astype(np.uint8).tobytes()).decode('utf-8')
        return v

    def get_mask(self) -> Optional[np.ndarray]:
        if self.mask is None:
            return None
        raw = base64.b64decode(self.mask)
        return np.frombuffer(raw, dtype=np.uint8).reshape(self.n_images, self.height, self.width)
