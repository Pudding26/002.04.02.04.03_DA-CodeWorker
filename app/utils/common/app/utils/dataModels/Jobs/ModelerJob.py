from __future__ import annotations
from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, ConfigDict

# ✅ Import job infrastructure
from app.utils.common.app.utils.SQL.models.jobs.orm_WorkerJobs import orm_WorkerJobs
from app.utils.common.app.utils.dataModels.Jobs.BaseJob import BaseJob
from app.utils.common.app.utils.dataModels.Jobs.JobEnums import JobStatus, JobKind

# ✅ Import configuration sub-models
from app.utils.common.app.utils.dataModels.configs.scaling import ScalingConfig
from app.utils.common.app.utils.dataModels.configs.metricModelling import DimReductionCfg, BinningCfg


# ─────────────────────────────────────────────────────────────────────────────
# STEP STATUS ENUM
# ─────────────────────────────────────────────────────────────────────────────

StepStatus = Union[
    Literal["passed"],  # Normal success
    str                 # Failure or explanation reason
]

# ─────────────────────────────────────────────────────────────────────────────
# FAIL TRAIL STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

class ModelerFailTrail(BaseModel):
    """
    FailTrail for tracking step-wise pass/fail states.
    Allows flexible nested structure while exposing mark().
    """
    preprocessing: Dict[str, Any] = {}
    modelling: Dict[str, Any] = {}
    validation: Dict[Any, Any] = {}

    def mark(self, section: str, step_name: str, status: Literal["passed", "failed", "skipped"], error: Optional[str] = None):
        """
        Backward-compatible flat marker for legacy stages.
        e.g., fail_trail.mark("modelling", "step_02", "failed")
        """
        section_dict = getattr(self, section, None)
        if isinstance(section_dict, dict):
            section_dict[step_name] = {"status": status, "error": error}
        else:
            setattr(self, section, {step_name: {"status": status, "error": error}})

    def mark_validation(
        self,
        bootstrap: int,
        label: str,
        frac: float,
        model: str,
        status: Literal["passed", "failed", "skipped"],
        error: Optional[str] = None
    ):
        """
        Nested fail marking for new validator system
        e.g., fail_trail.mark_validation(0, "genus", 0.1, "rf", "passed")
        """
        self.validation \
            .setdefault(bootstrap, {}) \
            .setdefault(label, {}) \
            .setdefault(frac, {})[model] = {
                "status": status,
                "error": error,
            }

    def __getitem__(self, key):
        return getattr(self, key)


# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAPPING CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class BootstrappingConfig(BaseModel):
    """
    Configuration for running N bootstrap resampling iterations on the input data.
    """
    enabled: bool = False                # Activate bootstrapping
    n_iterations: int = 10              # How many bootstrap replicates to run
    n_samples: Optional[int] = None     # Optional override for sample size (defaults to len(data))


# ─────────────────────────────────────────────────────────────────────────────
# RESAMPLING CONFIGURATION (SIMPLIFIED)
# ─────────────────────────────────────────────────────────────────────────────

class RandomSamplerConfig(BaseModel):
    scope: str
    sampling_strategy: str
    random_state: Optional[int] = 42
    mode: Optional[Literal["under", "over", "hybrid"]] = "over"
    strategy: Optional[Literal["mean", "median", "min", "max"]] = "median"

class SMOTESamplerConfig(BaseModel):
    scope: str
    k_neighbors: int = 5
    sampling_strategy: str
    random_state: Optional[int] = 42

class ResamplingConfig(BaseModel):
    method: Literal["RandomSampler", "SMOTESampler"]
    RandomSampler: Optional[RandomSamplerConfig] = None
    SMOTESampler: Optional[SMOTESamplerConfig] = None

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class PreProcessingAttributes(BaseModel):
    """
    Configuration block controlling data preprocessing steps before modeling.
    """
    scaling: Optional[ScalingConfig] = Field(default_factory=ScalingConfig)
    resampling: Optional[ResamplingConfig] = None
    bootstrapping: Optional[BootstrappingConfig] = Field(default_factory=BootstrappingConfig)

# ─────────────────────────────────────────────────────────────────────────────
# METRIC MODELING CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class MetricModelAttributes(BaseModel):
    """
    Configuration block controlling feature binning and dimensionality reduction.
    """
    binning_cfg: Optional[BinningCfg] = Field(default_factory=BinningCfg)
    dim_reduction_cfg: Optional[DimReductionCfg] = Field(default_factory=DimReductionCfg)

# ─────────────────────────────────────────────────────────────────────────────
# JOB INPUT (INSTRUCTIONS)
# ─────────────────────────────────────────────────────────────────────────────

class ModelerJobInput(BaseModel):
    """
    Contains all pipeline instructions (preprocessing + modeling), 
    and basic job-level identifiers.
    """
    stackIDs: List[str]                                 # Required input samples

    preProcessing_instructions: PreProcessingAttributes = Field(
        default_factory=PreProcessingAttributes,
        description="Preprocessing config (scaling, resampling, bootstrapping)"
    )
    metricModel_instructions: MetricModelAttributes = Field(
        default_factory=MetricModelAttributes,
        description="Model configuration (binning, dimensionality reduction)"
    )

    job_No: Optional[int] = None
    preProcessingNo: str                                # Preset ID
    metricModelNo: str                                  # Preset ID

    scope: Optional[str] = None                         # Taxonomic scope for resampling (e.g., species)
    index_col: Optional[int] = None                     # Encoded label column (for classification)

    bootstrap_iteration: int = 0                        # 0 = normal run; 1-N = bootstrapped replicate
    fail_trail: Optional[ModelerFailTrail] = Field(default_factory=ModelerFailTrail)

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ─────────────────────────────────────────────────────────────────────────────
# JOB ATTRIBUTES (STATE + RESULTS)
# ─────────────────────────────────────────────────────────────────────────────

class ModelerAttrs(BaseModel):
    """
    Contains the state of the job during/after execution:
    raw inputs, transformed matrices, modeling output, validation results.
    """
    raw_data: Optional[Any] = None                      # Raw pandas.DataFrame before processing
    preProcessed_data: Optional[Any] = None             # Preprocessed array (cupy/numpy)
    data_num: Optional[Any] = None                      # Numeric features ready for modeling
    model_results: Optional[Any] = None                 # Pandas DF from model stage

    encoder: Optional[Any] = None                       # Column name → index encoder dict
    colname_encoder: Optional[List[str]] = None         # Index → column name list

    engineered_data: Optional[Any] = None               # Output of feature engineering step
    multi_pca_results: Optional[Any] = None             # Dict[frac → PCA result]
    blacklist: Optional[List[str]] = None               # Feature names excluded from modeling
    dropped_fraction: Optional[float] = None            # % of rows dropped due to quality filters

    featureClusterMap: Optional[Any] = None             # UMAP / cluster visualization data
    result_df: Optional[Any] = None                     # Final metrics table
    results_cupy: Optional[Any] = None                  # Intermediate scores (GPU-based)

    uniques: Optional[Dict[str, int]] = None            # Class counts for labels
    bin_dict: Optional[Dict[str, Any]] = None           # Binning results (if binning applied)
    dim_red_dict: Optional[Dict[str, Any]] = None       # Dimensionality reduction result stored in a dict

    validation_results_dict: Dict[str, Any] = Field(default_factory=dict)  # Validation results per bootstrap
    validation_results_df: Optional[Any] = None  # Pandas DataFrame of validation results
# ─────────────────────────────────────────────────────────────────────────────
# MAIN JOB CLASS
# ─────────────────────────────────────────────────────────────────────────────

class ModelerJob(BaseJob):
    """
    Top-level modeling job object passed through the pipeline. 
    Orchestrator executes the job using its `.input` and updates `.attrs` and `.stats`.
    """
    job_type: JobKind = JobKind.MODELER
    orm_model = orm_WorkerJobs
    status: JobStatus = JobStatus.READY.value

    input: ModelerJobInput                              # Pipeline config + scope
    attrs: ModelerAttrs                                 # Data + results
    stats: Dict[str, Any] = Field(default_factory=dict) # Time, dimensions, accuracy, etc.
    context: Optional[Dict[str, Any]] = None            # Summary of pipeline steps (for UI/logs)

    model_config = ConfigDict(extra="forbid")
