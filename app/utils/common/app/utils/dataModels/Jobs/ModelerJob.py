from __future__ import annotations
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict

from app.utils.common.app.utils.SQL.models.jobs.orm_WorkerJobs import orm_WorkerJobs
from app.utils.common.app.utils.dataModels.Jobs.BaseJob import BaseJob
from app.utils.common.app.utils.dataModels.Jobs.JobEnums import JobStatus, JobKind

# ✅ Import config attributes from your structured modules
from app.utils.common.app.utils.dataModels.configs.scaling import ScalingConfig
from app.utils.common.app.utils.dataModels.configs.resampling import ResamplingConfig
from app.utils.common.app.utils.dataModels.configs.clustering import ClusterConfig
from app.utils.common.app.utils.dataModels.configs.metricModelling import DimReductionCfg, BinningCfg



class ModelerJob(BaseJob):
    job_type: JobKind = JobKind.MODELER
    orm_model = orm_WorkerJobs
    status: JobStatus = JobStatus.READY.value
    input: ModelerJobInput
    attrs: ModelerAttrs
    stats: Dict[str, Any] = Field(default_factory=dict)
    context: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="forbid")

class PreProcessingAttributes(BaseModel):
    scaling: Optional[ScalingConfig] = Field(default_factory=ScalingConfig)
    resampling: Optional[ResamplingConfig] = Field(default_factory=ResamplingConfig)
    method: Literal["scaling", "resampling", None] = None


class MetricModelAttributes(BaseModel):
    binning_cfg: Optional[BinningCfg] = Field(default_factory=BinningCfg)
    dim_reduction: Optional[DimReductionCfg] = Field(default_factory=DimReductionCfg)


class ModelerJobInput(BaseModel):
    stackIDs: List[str]

    preProcessing_instructions: PreProcessingAttributes = Field(
        default_factory=PreProcessingAttributes,
        description="Preprocessing config (scaling, normalization)"
    )
    metricModel_instructions: MetricModelAttributes = Field(
        default_factory=MetricModelAttributes,
        description="Model selection, CV, etc."
    )

    job_No: Optional[int] = None
    preProcessingNo: str
    metricModelNo: str
    scope: Optional[str] = None # Defines the lowest taxonmic hierachy that is used during resampling and modelling for clustering the data
    index_col: Optional[int] = None  # Integer represantiation of the index col in the encoded data

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ModelerAttrs(BaseModel):
    raw_data: Optional[Any] = None  # pandas.DataFrame
    preProcessed_data: Optional[Any] = None  # cupy DataFrame or np/cupy array
    model_results: Optional[Any] = None  # pandas.DataFrame
    data_num: Optional[Any] = None  # cupy DataFrame or np/cupy array
    encoder: Optional[Any] = None
    engineered_data: Optional[Any] = None  # cupy DataFrame or np/cupy array
    multi_pca_results: Optional[Any] = None # A dictionary with PCA results for each bin. The keys are the percentage of retained cols per bin.
    blacklist: Optional[List[str]] = None  # List of features that are not used in the model
    dropped_fraction: Optional[float] = None  # Retained fraction of rows


    featureClusterMap: Optional[Any] = None  # Dict with cluster labels and their features
    result_df: Optional[Any] = None  # Results of the model validation
    results_cupy: Optional[Any] = None  # Results of the model validation in cupy format
    uniques: Optional[Dict[str, int]] = None  # Unique counts per scope
