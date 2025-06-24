# feature_extractor.py
"""
Vectorised feature extractor with optional GPU acceleration.
===========================================================

* CPU path uses scikit-image (no Python loops).
* GPU path uses cuCIM + CuPy (works on any CUDA 11.8+ device).
* Public API is identical to your original extractor.

Usage
-----
>>> from feature_extractor import FeatureExtractor
>>> fe   = FeatureExtractor()
>>> df   = fe.apply_one(binary_mask)               # CPU
>>> df_g = fe.apply_one(binary_mask, use_gpu=True) # GPU (needs cuCIM / CuPy)
"""

from __future__ import annotations
from typing import Dict, Any, List

import numpy as np
import pandas as pd


class FeatureExtractor:
    """
    Extract geometric and topological features from a 2-D segmentation mask.

    Parameters
    ----------
    base_props : list[str], optional
        Which scikit-image properties to request.  Hu moments are included per
        default so no Python loop is needed.
    """

    DEFAULT_PROPS: List[str] = [
        "label",
        "area",
        "perimeter",
        "equivalent_diameter",
        "major_axis_length",
        "minor_axis_length",
        "eccentricity",
        "orientation",
        "extent",
        "solidity",
        "centroid",
        "euler_number",
        "moments_hu",   # 7 Hu moments → columns moments_hu-0 … moments_hu-6
        "bbox",         # needed for compactness
    ]

    def __init__(self, base_props: List[str] | None = None):
        self.base_props = base_props or self.DEFAULT_PROPS

    # ──────────────────────────────────────────────────────────────────────
    # main entry point
    # ──────────────────────────────────────────────────────────────────────
    def apply_one(
        self,
        mask: np.ndarray,
        *,
        connectivity: int = 2,
        use_gpu: bool = False,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        mask : np.ndarray
            2-D binary (bool / uint8) mask.
        connectivity : int, default = 2
            1 = 4-connected, 2 = 8-connected component labeling.
        use_gpu : bool, default = False
            True → run on GPU via cuCIM + CuPy.

        Returns
        -------
        pd.DataFrame
            Wide-format table, one row per object.
        """
        if use_gpu:
            labelled, props_dict = self._gpu_props(mask, connectivity)
        else:
            labelled, props_dict = self._cpu_props(mask, connectivity)

        df = pd.DataFrame(props_dict, dtype="float32")

        # ── derived metrics (vectorised) ────────────────────────────────
        perim = df["perimeter"].replace(0, np.nan)
        maj   = df["major_axis_length"].replace(0, np.nan)
        minr  = df["minor_axis_length"].replace(0, np.nan)

       # df["roundness"]    = 4 * np.pi * df["area"] / (perim ** 2) #TODO
        df["aspect_ratio"] = maj / minr
        #df["circularity"]  = (perim ** 2) / (4 * np.pi * df["area"]) #TODO

        bbox_w = df["bbox-3"] - df["bbox-1"]
        bbox_h = df["bbox-2"] - df["bbox-0"]
        bbox_a = (bbox_w * bbox_h).replace(0, np.nan)
        df["compactness"] = df["area"] / bbox_a

        return df

    # ──────────────────────────────────────────────────────────────────────
    # internal helpers
    # ──────────────────────────────────────────────────────────────────────
    def _cpu_props(
        self, mask: np.ndarray, connectivity: int
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        from skimage.measure import label, regionprops_table

        labelled = label(mask.astype(np.uint8), connectivity=connectivity)
        props = regionprops_table(
            labelled,
            properties=self.base_props,
            cache=True,
        )
        return labelled, props

    def _gpu_props(
        self, mask: np.ndarray, connectivity: int
    ) -> tuple["cupy.ndarray", Dict[str, Any]]:
        import cupy as cp
        from cucim.skimage.measure import label, regionprops_table

        xp = cp
        labelled = label(xp.asarray(mask).astype(xp.uint8),
                         connectivity=connectivity)
        props_gpu = regionprops_table(
            labelled,
            properties=self.base_props,
            cache=True,
        )
        # convert CuPy → NumPy for Pandas
        props = {k: xp.asnumpy(v) for k, v in props_gpu.items()}
        # free VRAM early
        xp.cuda.runtime.deviceSynchronize()
        xp.get_default_memory_pool().free_all_blocks()
        return labelled, props
