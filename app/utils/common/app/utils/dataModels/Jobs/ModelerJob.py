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
    Also records last failure/error for quick retrieval.
    """
    preprocessing: Dict[str, Any] = {}
    modelling: Dict[str, Any] = {}
    validation: Dict[Any, Any] = {}
    last_fail: Optional[Dict[str, Any]] = None

    def mark(
        self,
        section: str,
        step_name: str,
        status: Literal["passed", "failed", "skipped", "error"],
        error: Optional[str] = None
    ):
        """
        Legacy-compatible flat marker.
        Tracks status + stores last_fail if failed/error.
        """
        section_dict = getattr(self, section, None)
        entry = {"status": status, "error": error}

        if isinstance(section_dict, dict):
            section_dict[step_name] = entry
        else:
            setattr(self, section, {step_name: entry})

 
        self.last_fail = {
            "section": section,
            "step": step_name,
            "status": status,
            "error": error
        }

    def mark_validation(
        self,
        fold_no: int,
        bootstrap_no: int,
        label: str,
        frac: float,
        model: str,
        status: Literal["passed", "failed", "skipped", "error"],
        error: Optional[str] = None
    ):
        """
        Nested marker for validator with fold/boostrap context.
        Also updates last_fail if failed/error.
        """
        self.validation \
            .setdefault(fold_no, {}) \
            .setdefault(bootstrap_no, {}) \
            .setdefault(label, {}) \
            .setdefault(frac, {})[model] = {
                "status": status,
                "error": error
            }

        self.last_fail = {
            "section": "validation",
            "fold_no": fold_no,
            "bootstrap_no": bootstrap_no,
            "label": label,
            "frac": frac,
            "model": model,
            "status": status,
            "error": error
        }


    def get_last_fail_reason(self) -> Optional[str]:
        if self.last_fail:
            return self.last_fail.get("error")
        return None
    
    def get_last_status(self) -> Optional[str]:
        if self.last_fail:
            return self.last_fail.get("status")
        return None
    


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

    outer_fold: Optional[int] = None                    # Outer fold number for cross-validation

    bootstrap_iteration: Optional[int] = None            # 0 = normal run; 1-N = bootstrapped replicate

    initial_cols_count: Optional[int] = None              # Initial column count
    initial_rows_count: Optional[int] = None                     # Initial number of rows before any processing

    validation_cols: Optional[float] = None  # Columns used for validation (e.g., labels)
    validation_rows: Optional[float] = None  # Number of rows used for validation
    
    fail_trail: Optional[ModelerFailTrail] = Field(default_factory=ModelerFailTrail)

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ─────────────────────────────────────────────────────────────────────────────
# JOB ATTRIBUTES (STATE + RESULTS)
# ─────────────────────────────────────────────────────────────────────────────

class ModelerAttrs(BaseModel):
    """
    State container for a modeler job during and after execution.

    This object holds all intermediate and final state for a modeling pipeline run:
    from raw inputs, through preprocessing and dimensionality reduction,
    to model training, validation, and results.

    Fields:
    --------
    raw_data : Optional[Any]
        Original raw input data as a pandas.DataFrame, before any GPU preprocessing.

    encoder : Optional[Any]
        Single source of truth for encoding mappings:
        ▸ encoder.cols : dict[str → int] — maps column names to column indices in all numeric matrices.
        ▸ encoder.vals : dict[str → cudf.Series] — maps column names to their encoded categories (for inverse mapping if needed).
        Preserved consistently across all sub-jobs to ensure reproducibility.

    data_num : Optional[Any]
        Current working dataset as a GPU-compatible numeric matrix (CuPy or NumPy ndarray).
        ▸ In the refactored pipeline: always holds the **training partition after train/test split**.
        ▸ Acts as input for sampler, bootstrapper, PCA, and downstream modeler modules.
        ▸ All columns indexed according to `encoder.cols`.

    data_test : Optional[Any]
        Test (holdout) dataset as a GPU-compatible numeric matrix.
        ▸ Untouched by sampling/bootstrapping.
        ▸ Same structure and encoding as `data_num`.
        ▸ Used by validator after applying PCA transform from training partition.

    data_train : Optional[Any]
        Optional explicit training dataset.
        Encoded TRAIN partition after splitter.
        The main working dataset for sampler, bootstrapper, PCA, etc.
    """
    raw_data: Optional[Any] = None                      # Raw pandas.DataFrame before processing
    
    encoder: Optional[Any] = None                       # Column name → index encoder dict
    
    data_num: Optional[Any] = None                      # Numeric features ready for modeling
    
    
    data_train: Optional[Any] = None                   # Training data (cupy/numpy)
    data_test: Optional[Any] = None                    # Test data (cupy/numpy)
    
    
    
    
    
    preProcessed_data: Optional[Any] = None             # Preprocessed array (cupy/numpy)
    model_results: Optional[Any] = None                 # Pandas DF from model stage

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
