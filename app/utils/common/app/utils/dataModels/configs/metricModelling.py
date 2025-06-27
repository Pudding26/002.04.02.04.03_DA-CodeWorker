from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal, Optional




class MetricModellingConfig(BaseModel):
    """
    Configuration for metric modeling strategies.
    Define the parameters and methods for metric modeling.
    """
    submethod: Optional[Literal["PCA", "t-SNE", "UMAP"]] = Field(default="none", description="Top-level metric modeling method selector")
    PCA: Optional[PCAParams] = Field(default=None, description="Parameters for PCA")
    t_SNE: Optional[tSNEParams] = Field(default=None, description="Parameters for t-SNE")
    UMAP: Optional[UMAPParams] = Field(default=None, description="Parameters for UMAP")
