from __future__ import annotations
from typing import List, Optional, Union, Dict, Any
from uuid import uuid4
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field

import base64
import numpy as np
from pydantic import BaseModel, field_validator

#ORM
from app.utils.SQL.models.jobs.orm_JobLink import orm_JobLink



from app.utils.dataModels.Jobs.BaseJob import BaseJob
from app.utils.dataModels.Jobs.DoEJob import DoEJob
from app.utils.dataModels.Jobs.ExtractorJob import ExtractorJobInput


from app.utils.dataModels.Jobs.JobEnums import JobStatus, JobKind, RelationState


from app.utils.SQL.models.jobs.orm_WorkerJobs import orm_WorkerJobs

from uuid import UUID

from sqlalchemy import event, delete, insert, update, select, func
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Mapper




class SegmenterJob(BaseJob):
    
    
    job_type: JobKind = JobKind.SEGMENTER
    orm_model = orm_WorkerJobs
    status: JobStatus = JobStatus.READY.value
    input: SegmenterJobInput
    attrs: SegmenterAttrs
    

    model_config = ConfigDict(extra="forbid")


class SegmenterJobInput(BaseModel):
    hdf5_path: str = "data/productionData/primaryData.hdf5"
    src_file_path: str
    
    dest_file_path_GS: str
    dest_file_path_FF: str
    
    dest_stackID_FF: str
    dest_stackID_GS: str

    dest_FilterNo: str
    filter_instructions: Dict[str, Any] = Field(default_factory=dict)

    

    image_GS: Optional[Any] = None
    image_FF: Optional[Any] = None


    job_No: Optional[int] = None



    model_config = ConfigDict(arbitrary_types_allowed=True)

class SegmenterAttrs(BaseModel):
    attrs_raw: Dict[str, Any] = Field(default_factory=dict)
    attrs_FF: Dict[str, Any] = Field(default_factory=dict)
    attrs_GS: Dict[str, Any] = Field(default_factory=dict)
    features_df: Optional[Any] = None  # Placeholder for DataFrame or similar structure
    segmentation_mask_raw: Optional[Any] = None  # Placeholder for segmentation mask data
    extractorJobinput: Optional[ExtractorJobInput] = None
