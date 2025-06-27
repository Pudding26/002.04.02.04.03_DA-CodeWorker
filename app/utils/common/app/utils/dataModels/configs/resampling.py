from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal, Optional


# ------------------------------
# Overall Wrapper
# ------------------------------


# ------------------------------
# Oversampling Wrapper
# ------------------------------

class OversamplingConfig(BaseModel):
    """
    Configuration for standardization methods, such as MinMax or StandardScaler.
    Choose a sub-method and provide related parameters.
    """
    subsubmethod: Literal[None] = None



# ------------------------------
# Oversampling Options
# ------------------------------


# ------------------------------
# Undersampling Wrapper
# ------------------------------

class UndersamplingConfig(BaseModel):
    """
    Configuration for standardization methods, such as MinMax or StandardScaler.
    Choose a sub-method and provide related parameters.
    """
    subsubmethod: Literal[None] = None


# ------------------------------
# Undersampling Options
# ------------------------------



# ------------------------------
# Bootstrapping Wrapper
# ------------------------------


class BootstrappingConfig(BaseModel):
    """
    Configuration for Bootstraping methods. Choose a sub-method and provide related parameters.
    """
    subsubmethod: Literal[None] = None




# ------------------------------
# Bootstrapping Options
# ------------------------------



# ------------------------------
# Overall Wrapper
# ------------------------------

class ResamplingConfig(BaseModel):
    """
    Top-level configuration for resampling strategies.
    Select the method and configure its parameters below.
    """
    submethod: Literal["bootstrapping", "oversampling", "undersampling", None] = None
    bootstrapping: Optional[BootstrappingConfig] = Field(default_factory=BootstrappingConfig, description="Bootstrapping configuration")
    oversampling: Optional[OversamplingConfig] = Field(default_factory=OversamplingConfig, description="Oversampling configuration")
    undersampling: Optional[UndersamplingConfig] = Field(default_factory=UndersamplingConfig, description="Undersampling configuration")