from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, List, Optional
from typing_extensions import Literal

# ─────────── BINNING ───────────────────────────────────────────────
class ExplicitBinning(BaseModel):
    presets:  Optional[Dict[str, List[str]]] = None
    clusters: Dict[str, List[str]]
    agg_func: Literal["mean", "median", "sum", "none"] = "mean"

class ImplicitBinning(BaseModel):
    patterns:    Dict[str, List[str]]
    ignore_case: bool = True
    agg_func:    Literal["mean", "median", "sum", "none"] = "mean"

class BinningCfg(BaseModel):
    strategy: Literal["none", "explicit", "implicit"]
    explicit: Optional[ExplicitBinning] = None
    implicit: Optional[ImplicitBinning] = None

# ─────────── LINEAR (PCA) ──────────────────────────────────────────
class PCAParams(BaseModel):
    n_components: float | int = Field(0.95, ge=0)
    whiten: bool = False

class LinearFamily(BaseModel):
    submethod: Literal["PCA"]
    PCA: Optional[PCAParams] = None

# ─────────── MANIFOLD (t-SNE / UMAP) ───────────────────────────────
class TSNEParams(BaseModel):
    n_components: int = 2
    perplexity:   float = Field(30, gt=0)
    learning_rate:float = Field(200.0, gt=0)
    n_iter:       int = 1_000
    random_state: Optional[int] = None

class UMAPParams(BaseModel):
    n_components: int = 2
    n_neighbors:  int = 15
    min_dist:     float = 0.1
    metric:       str   = "euclidean"

class ManifoldFamily(BaseModel):
    submethod: Literal["tSNE", "UMAP"]
    tSNE: Optional[TSNEParams] = None
    UMAP: Optional[UMAPParams] = None

# ─────────── ENCODER (Seg / Img) ───────────────────────────────────
class AutoencoderSegParams(BaseModel):
    encoder_filters: List[int]
    code_dim:  int
    dropout:   float = 0.0
    loss:      Literal["dice", "bce", "jaccard"] = "bce"

class AutoencoderImgParams(BaseModel):
    backbone:   str = "resnet34"
    code_dim:   int = 128
    pretrained: bool = True
    dropout:    float = 0.0
    aug_strength: float = Field(0.0, ge=0, le=1)

class EncoderFamily(BaseModel):
    submethod: Literal["AutoencoderSeg", "AutoencoderImg"]
    AutoencoderSeg: Optional[AutoencoderSegParams] = None
    AutoencoderImg: Optional[AutoencoderImgParams] = None

# ─────────── MASTER DIM-RED Params ─────────────────────────────────
class DimReductionCfg(BaseModel):
    method: Literal["linear", "manifold", "encoder"]
    linear:    Optional[LinearFamily]    = None
    manifold:  Optional[ManifoldFamily]  = None
    encoder:   Optional[EncoderFamily]   = None

    @model_validator(mode="after")
    def check_chosen_block(self):
        m = self.method
        if m == "linear" and not self.linear:
            raise ValueError("linear block missing")
        if m == "manifold" and not self.manifold:
            raise ValueError("manifold block missing")
        if m == "encoder" and not self.encoder:
            raise ValueError("encoder block missing")
        return self


class MetricModellingConfig(BaseModel):
    name: str
    binning_cfg:   Optional[BinningCfg]    = None 
    dim_reduction: DimReductionCfg
