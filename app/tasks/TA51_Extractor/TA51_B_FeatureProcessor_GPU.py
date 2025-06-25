# TA30_C_SegmentationFeatureProcessor_GPU.py
from __future__ import annotations
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
import cudf
import cupy as cp


# ──────────────────────────────────────────────────────────────────────
def _flatten_value(val):
    if isinstance(val, (list, tuple, np.ndarray)):
        return val[0] if len(val) else np.nan
    return val


_STAT_FUNCS: Dict[str, Callable[[cudf.Series], float]] = {
    "mean": cudf.Series.mean,
    "std": cudf.Series.std,
    "min": cudf.Series.min,
    "max": cudf.Series.max,
    "median": cudf.Series.median,
    # "skew" handled separately
}


class TA51_B_FeatureProcessor_GPU:
    # ------------------------------------------------------------------
    # Hard-coded instructions
    # ------------------------------------------------------------------
    PERCENTILES = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                   55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

    BIN_FEATURES = ["area", "roundness", "eccentricity"]

    SUMMARY_FEATURES = [
        "area",
        "roundness",
        "eccentricity",
        "major_axis_length",
        "solidity",
        "orientation",
    ]

    STATS = ["mean", "std", "min", "max", "median", "skew"]

    UNITS = {
        "area": "px²",
        "roundness": "-",
        "eccentricity": "-",
        "major_axis_length": "px",
        "solidity": "-",
        "orientation": "degrees",
    }

    _META_COLS = ("stackID", "shotID", "position")

    def process(self, stackID: str, feature_tables: List[pd.DataFrame]) -> pd.DataFrame:
        if not feature_tables:
            return pd.DataFrame()

        dfs: List[pd.DataFrame] = []
        for idx, df in enumerate(feature_tables):
            # +1 to start from 1 instead of 0
            shot_id = f"{stackID}_{idx+1:03d}"
            df = df.copy()
            df["stackID"] = stackID
            df["shotID"] = shot_id
            dfs.append(self._process_single_table(df))

        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def _process_single_table(self, df_in: pd.DataFrame) -> pd.DataFrame:
        meta_vals = {col: df_in[col].iloc[0] for col in self._META_COLS}
        df = df_in.copy()
        for col in (*self.BIN_FEATURES, *self.SUMMARY_FEATURES):
            if col in df.columns:
                df[col] = df[col].map(_flatten_value)
        return self._bin_and_summarise_all(df, meta_vals)

    def _bin_and_summarise_all(self, df: pd.DataFrame, meta_vals: dict) -> pd.DataFrame:
        df_gpu = cudf.DataFrame(df)

        all_rows: List[Dict] = []
        for bin_feature in self.BIN_FEATURES:
            if bin_feature not in df_gpu.columns:
                continue
            all_rows.extend(self._bin_and_summarise(df_gpu, bin_feature, meta_vals))
        return pd.DataFrame(all_rows)

    def _bin_and_summarise(
        self, df_gpu: cudf.DataFrame, bin_feature: str, meta_vals: dict
    ) -> List[Dict]:
        series = df_gpu[bin_feature].dropna()
        if series.empty:
            return []

        q = len(self.PERCENTILES) - 1
        bins, cats = cudf.qcut(series, q=q, retbins=True, duplicates="drop")
        labels = self._make_bin_labels(cats)
        df_gpu = df_gpu.assign(__bin=bins.cat.rename_categories(labels))

        long_rows = []
        for bin_label, sub in df_gpu.groupby("__bin"):
            obj_count = len(sub)
            if obj_count < 1:
                continue
            for feature in self.SUMMARY_FEATURES:
                if feature not in sub.columns:
                    continue
                for stat in self.STATS:
                    if stat == "skew":
                        col_cupy = sub[feature].to_cupy()
                        mean = cp.mean(col_cupy)
                        std = cp.std(col_cupy)
                        val = float(cp.mean((col_cupy - mean) ** 3) / std**3)
                    else:
                        val = float(_STAT_FUNCS[stat](sub[feature]))

                    long_rows.append(
                        {
                            **meta_vals,
                            "bin_type": f"by_{bin_feature}",
                            "percentile_bin": bin_label,
                            "feature_name": feature,
                            "stat_type": stat,
                            "feature_value": val,
                            "unit": self.UNITS.get(feature, "-"),
                            "object_count": int(obj_count),
                        }
                    )
        return long_rows

    @staticmethod
    def _make_bin_labels(categories) -> List[str]:
        n = len(categories)
        if n == 0:
            return []
        step = 100 / n
        return [f"p{int(round(i*step))}-p{int(round((i+1)*step))}" for i in range(n)]
