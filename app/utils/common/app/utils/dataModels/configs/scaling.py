from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal, Optional



# ------------------------------
# Standartization Wrapper
# ------------------------------

class StandardizationConfig(BaseModel):
    """
    Configuration for standardization methods, such as MinMax or StandardScaler.
    Choose a sub-method and provide related parameters.
    """
    subsubmethod: Optional[Literal[None , "MinMaxScaler", "StandardScaler"]] = Field(default=None, description="Sub-method to apply for standardization")
    MinMaxScaler: Optional[MinMaxScalerParams] = Field(default=None, description="Parameters for Min-Max scaling")
    StandardScaler: Optional[StandardScalerParams] = Field(default=None, description="Parameters for Standard scaling")

# ------------------------------
# Standartization Options
# ------------------------------

class MinMaxScalerParams(BaseModel):
    """
    Parameters for MinMaxScaler. Rescales features to a defined interval.
    """
    min: float = Field(default=0.0, description="Minimum value after scaling")
    max: float = Field(default=1.0, description="Maximum value after scaling")
    #clip: bool = Field(default=False, description="Whether to clip values outside the [min, max] range") #ONLY FOR CPU


class StandardScalerParams(BaseModel):
    """
    Parameters for StandardScaler. Centers features and scales them to unit variance.
    """
    with_mean: bool = Field(default=True, description="Center data before scaling")
    with_std: bool = Field(default=True, description="Scale data to unit variance")


# ------------------------------
# Normalization Wrapper
# ------------------------------


class NormalizationConfig(BaseModel):
    """
    Configuration block for normalization strategies. Currently supports only standard L1/L2/max norms.
    """
    subsubmethod: Optional[Literal[None, "StandardNormalizer"]] = Field(default=None, description="Normalization method to apply")
    StandardNormalizer: Optional[StandardNormalizerParams] = Field(default=None, description="Parameters for normalization")

# ------------------------------
# Normalization Options
# ------------------------------

class StandardNormalizerParams(BaseModel):
    """
    Parameters for row-wise normalization using l1, l2, or max norm.
    Useful for algorithms sensitive to vector magnitude.
    """
    norm: Literal["l1", "l2", "max"] = Field(default="l2", description="Type of norm to apply")
    axis: int = Field(default=1, description="Axis to normalize along")


# ------------------------------
# Overall Wrapper
# ------------------------------


class ScalingConfig(BaseModel):
    """
    Top-level entry for scaling logic.
    Choose a method and define its parameters through nested structures.
    """
    submethod: Optional[Literal["none", "standardization", "normalization"]] = Field(default="none", description="Top-level scaling method selector")
    standardization: Optional[StandardizationConfig] = Field(default_factory=StandardizationConfig, description="Standardization strategy")
    normalization: Optional[NormalizationConfig] = Field(default_factory=NormalizationConfig, description="Normalization strategy")

