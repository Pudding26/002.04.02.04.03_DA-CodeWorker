from __future__ import annotations
from typing import Sequence, List, Tuple
import cupy as cp, cudf, numpy as np
from cucim.skimage.measure import label, regionprops_table

class TA51_A_FeatureExtractor_GPU:
    _PROPS = [
        "label",
        "area",
        "eccentricity",
        "orientation",
        "major_axis_length",
        "solidity",
        "centroid",
    ]
    _CONNECTIVITY = 2

    def __init__(self, base_props: List[str] | None = None):
        self.base_props = base_props or self._PROPS

    def apply_batch_gpu(self, masks: Sequence[np.ndarray | cp.ndarray]) -> List[cudf.DataFrame]:
        if isinstance(masks, np.ndarray) and masks.ndim == 3:
            masks = list(masks)
        if not masks:
            return []

        gpu_masks = [cp.asarray(m, dtype=cp.uint8) for m in masks]

        # Get max height & width
        H = max(m.shape[-2] for m in gpu_masks)
        W = max(m.shape[-1] for m in gpu_masks)

        # Stack into one long 2D canvas vertically
        canvas = cp.zeros((len(gpu_masks) * H, W), dtype=cp.uint8)
        for i, m in enumerate(gpu_masks):
            h, w = m.shape[-2:]
            canvas[i * H:i * H + h, :w] = m

        # Label in one shot
        lbl = label(canvas, connectivity=self._CONNECTIVITY)
        props = regionprops_table(lbl, properties=self.base_props)

        df_full = cudf.DataFrame({k: v for k, v in props.items()})

        # Compute which slice each region came from
        df_full["slice_id"] = cp.floor(df_full["centroid-0"] / H).astype(cp.int32)

        # Drop centroid columns
        df_full = df_full.drop(
            columns=[c for c in df_full.columns if c.startswith("centroid")]
        )

        # Split into one dataframe per mask
        dfs = []
        for i in range(len(gpu_masks)):
            dfs.append(
                df_full[df_full["slice_id"] == i]
                .drop(columns="slice_id")
                .reset_index(drop=True)
            )
        return dfs
