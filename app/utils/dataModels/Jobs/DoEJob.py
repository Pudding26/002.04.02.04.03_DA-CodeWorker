from __future__ import annotations

from pydantic import BaseModel, ConfigDict, conlist, field_validator
from typing import List, TypeAlias, Literal, Union, Iterable, ClassVar
import pandas as pd
import math

from sqlalchemy.orm import relationship




from app.utils.dataModels.Jobs.BaseJob import BaseJob
from app.utils.dataModels.Jobs.JobEnums import JobStatus, JobKind

from app.utils.SQL.models.jobs.orm_DoEJobs import orm_DoEJobs



class DoEJob(BaseJob):
    job_type : JobKind = JobKind.DOE.value

    orm_model = orm_DoEJobs

    status: JobStatus = JobStatus.READY.value
    provider_status: JobStatus = JobStatus.READY.value
    segmenter_status: JobStatus = JobStatus.BLOCKED.value
    extractor_status: JobStatus = JobStatus.BLOCKED.value
    modeler_status: JobStatus = JobStatus.BLOCKED.value
    validator_status: JobStatus = JobStatus.BLOCKED.value

    doe_config    : DOE_config

    model_config = ConfigDict(populate_by_name=True)

    ## inside DoEJob model class:
    #child_links: ClassVar = relationship(
    #    "JobLink",
    #    back_populates="parent",
    #    cascade="all, delete-orphan",
    #)



    def to_filter_row(self) -> dict:
        """Return a flat dict containing all filter-relevant fields and job_uuid."""
        primary = self.doe_config.primary_data
        return {
            "job_uuid":        str(self.job_uuid),
            "sourceNo":        primary.sourceNo,
            "woodType":        primary.woodType,
            "family":          primary.family,
            "genus":           primary.genus,
            "species":         primary.species,
            "view":            primary.view,
            "lens":            primary.lens,
            "maxShots":        primary.maxShots,
            "noShotsRange":    primary.noShotsRange,
            "filterNo":        self.doe_config.segmentation.filterNo,
            "secondaryDataBins": self.doe_config.secondary_data.secondaryDataBins,
            "preProcessingNo":   self.doe_config.preprocessing.preProcessingNo,
            "metricModelNo":     self.doe_config.modeling.metricModelNo,
        }

    @classmethod
    def to_filter_df(cls, jobs: Iterable["DoEJob"]) -> pd.DataFrame:
        """
        Convert multiple DoEJob objects into a flat DataFrame of filter fields.
        Result uses job_uuid as index.
        """
        rows = [job.to_filter_row() for job in jobs]
        return pd.DataFrame(rows).set_index("job_uuid")
    

    def to_sql_row(self) -> dict:
        def _clean_nans(obj):
            if isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return None
                return obj
            elif isinstance(obj, dict):
                return {k: _clean_nans(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_clean_nans(item) for item in obj]
            return obj

        raw_dict = self.model_dump(mode="json")
        clean_dict = _clean_nans(raw_dict)

        # Top-level fields
        row = {
            "job_uuid": self.job_uuid,
            "job_type": self.job_type,
            "status": self.status,
            "provider_status": self.provider_status,
            "segmenter_status": self.segmenter_status,
            "extractor_status": self.extractor_status,
            "modeler_status": self.modeler_status,
            "validator_status": self.validator_status,
            "attempts": self.attempts,
            "next_retry": self.next_retry,
            "created": self.created,
            "updated": self.updated,
            "parent_job_uuids": self.parent_job_uuids,
            "payload": {
                "doe_config": clean_dict["doe_config"],

            }
        }

        return row



class DOE_config(BaseModel):
    primary_data   : PrimaryData
    segmentation   : SegmentationCfg
    secondary_data : SecondaryData
    preprocessing  : PreProcessingCfg
    modeling       : ModelingCfg


class PrimaryData(BaseModel):
    sourceNo        : List[str]
    woodType        : List[str] 
    family          : List[str]
    genus           : List[str]
    species         : List[str]
    view            : List[str]
    lens            : List[float]
    maxShots        : List[int]
    noShotsRange: List[List[int]]

    model_config = ConfigDict(extra="forbid")

    @field_validator("noShotsRange")
    def check_range_shape(cls, v: Union[List[List[int]], List[None]]) -> Union[List[List[int]], List[None]]:
        if v == [None]:
            return v  # wildcard accepted
        if not isinstance(v, list):
            raise ValueError(f"noShotsRange must be a list, got {type(v)}")
        for pair in v:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"Each noShotsRange entry must be a list of 2 integers, got: {pair}")
            if not all(isinstance(i, int) for i in pair):
                raise ValueError(f"Invalid integers in noShotsRange: {pair}")
        return v




class SegmentationCfg(BaseModel):
    filterNo : List[str]
    model_config = ConfigDict(populate_by_name=True)

class SecondaryData(BaseModel):
    secondaryDataBins : List[str]
    model_config = ConfigDict(populate_by_name=True)

class PreProcessingCfg(BaseModel):
    preProcessingNo : List[str]
    model_config = ConfigDict(populate_by_name=True)

class ModelingCfg(BaseModel):
    metricModelNo : List[str]
    model_config = ConfigDict(populate_by_name=True)