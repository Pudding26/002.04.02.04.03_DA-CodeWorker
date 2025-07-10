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



# ─────────── MASTER DIM-RED Params ─────────────────────────────────
class DimReductionCfg(BaseModel):
    method: Literal["PCA", "t-SNE", "UMAP", "AutoencoderSeg", "AutoencoderImg"]
    PCA: Optional[PCAParams] = None
    t_SNE: Optional[TSNEParams] = None
    UMAP: Optional[UMAPParams] = None
    AutoencoderSeg: Optional[AutoencoderSegParams] = None
    AutoencoderImg: Optional[AutoencoderImgParams] = None




class BinningCfg(BaseModel):
    """
    Configuration for binning strategies.
    Choose a strategy and define related parameters.
    """
    strategy: Literal["none", "explicit", "implicit"] = "none"
    explicit: Optional[ExplicitBinning] = None
    implicit: Optional[ImplicitBinning] = None
    