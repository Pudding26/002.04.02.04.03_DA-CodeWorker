from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal, Optional, Union, List


# ------------------------------
# Oversampling Options
# ------------------------------

class RandomOverSamplerParams(BaseModel):
    """
    Parameters for random oversampling. Replicates minority samples to balance class sizes.
    """
    subsubmethod: Optional[Literal["RandomOverSampler"]] = "RandomOverSampler"
    target_size: Optional[int] = None
    random_state: Optional[int] = 42
    scope: Union[str, List[str]] = "shotID"


class SMOTESamplerParams(BaseModel):
    """
    Parameters for SMOTE-based oversampling using k-nearest neighbors to synthesize new samples.
    """
    subsubmethod: Optional[Literal["SMOTESampler"]] = "SMOTESampler"
    k_neighbors: Optional[int] = 5
    target_size: Optional[int] = None
    random_state: Optional[int] = 42
    scope: Union[str, List[str]] = "shotID"
    sampling_strategy: Union[str, float, dict] = "auto"


# ------------------------------
# Oversampling Wrapper
# ------------------------------

class OversamplingConfig(BaseModel):
    """
    Configuration for oversampling strategies.
    Choose a sub-method and define related parameters.
    """
    subsubmethod: Optional[Literal[None, "RandomOverSampler", "SMOTESampler"]] = None
    RandomOverSampler: Optional[RandomOverSamplerParams] = None
    SMOTESampler: Optional[SMOTESamplerParams] = None


# ------------------------------
# Undersampling Options
# ------------------------------

class RandomUnderSamplerParams(BaseModel):
    """
    Parameters for random undersampling. Downsamples majority classes.
    """
    subsubmethod: Optional[Literal["RandomUnderSampler"]] = "RandomUnderSampler"
    target_size: Optional[int] = None
    random_state: Optional[int] = 42
    scope: Union[str, List[str]] = "shotID"


# ------------------------------
# Undersampling Wrapper
# ------------------------------

class UndersamplingConfig(BaseModel):
    """
    Configuration for undersampling strategies.
    Choose a sub-method and define related parameters.
    """
    subsubmethod: Optional[Literal[None, "RandomUnderSampler"]] = None
    RandomUnderSampler: Optional[RandomUnderSamplerParams] = None


# ------------------------------
# Bootstrapping Options
# ------------------------------

class BootstrappingParams(BaseModel):
    """
    Parameters for sampling with replacement.
    """
    subsubmethod: Optional[Literal["BootstrapSampler"]] = "BootstrapSampler"
    random_state: Optional[int] = 42
    n_samples: Optional[int] = None
    scope: Union[str, List[str]] = "shotID"


# ------------------------------
# Bootstrapping Wrapper
# ------------------------------

class BootstrappingConfig(BaseModel):
    """
    Configuration for bootstrapping strategies.
    Choose a sub-method and define related parameters.
    """
    subsubmethod: Optional[Literal[None, "BootstrapSampler"]] = None
    BootstrapSampler: Optional[BootstrappingParams] = None


# ------------------------------
# Hybrid Sampling Options
# ------------------------------

class HybridSamplerParams(BaseModel):
    """
    Parameters for hybrid resampling (over + under to a shared target).
    """
    subsubmethod: Optional[Literal["HybridSampler"]] = "HybridSampler"
    target_size: Optional[int] = None
    strategy: Optional[Literal["median", "mean", "fixed"]] = "median"
    random_state: Optional[int] = 42
    scope: Union[str, List[str]] = "shotID"


# ------------------------------
# Hybrid Sampling Wrapper
# ------------------------------

class HybridResamplingConfig(BaseModel):
    """
    Configuration for hybrid sampling strategies.
    """
    subsubmethod: Optional[Literal[None, "HybridSampler"]] = None
    HybridSampler: Optional[HybridSamplerParams] = None


# ------------------------------
# Overall Wrapper
# ------------------------------

class ResamplingConfig(BaseModel):
    """
    Top-level configuration for resampling strategies.
    Select a method and configure its parameters.
    """
    submethod: Optional[Literal[
        "bootstrapping",
        "oversampling",
        "undersampling",
        "hybrid",
        "smote",
        None
    ]] = None

    bootstrapping: Optional[BootstrappingConfig] = Field(default_factory=BootstrappingConfig, description="Bootstrapping configuration")
    oversampling: Optional[OversamplingConfig] = Field(default_factory=OversamplingConfig, description="Oversampling configuration")
    undersampling: Optional[UndersamplingConfig] = Field(default_factory=UndersamplingConfig, description="Undersampling configuration")
    hybrid: Optional[HybridResamplingConfig] = Field(default_factory=HybridResamplingConfig, description="Hybrid (over + under) configuration")
