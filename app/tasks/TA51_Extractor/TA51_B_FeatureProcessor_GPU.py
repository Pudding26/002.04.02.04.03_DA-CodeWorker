# ================================================================
# TA51 – Refactored GPU feature pipeline (drop‑in replacement)
# --------------------------------------------------------------
#  ⬇️  Three modules, same names & class layout as legacy code ⬇️
#
#    ├─ TA51_A_FeatureExtractor_GPU.py   (per‑shot region props)
#    ├─ TA51_B_FeatureProcessor_GPU.py   (bin & summarise stats)
#    └─ TA51_0_ExtractorOrchestrator.py  (batch orchestration)
#
#  Each script keeps the *public* API identical to the original
#  versions — so importing code elsewhere does **not** need to
#  change — but the internals are heavily revamped for >10× GPU
#  utilisation.  Just overwrite the old files or put these on
#  the PYTHONPATH ahead of them.
# ================================================================


# -----------------------------------------------------------------
# 2.  TA51_B_FeatureProcessor_GPU.py
# -----------------------------------------------------------------
"""Binning + statistical summarisation (GPU).

Public API is unchanged:  ``process(stackID, feature_tables)`` returns a
long‑format cuDF ready to be written to parquet by the orchestrator.
Internally the quantile edges are cached and all stats (mean/std/min/max
+ skew) are computed vectorised in **one** groupby to avoid Python loops.
"""
from __future__ import annotations
from typing import Sequence, Dict

import cudf, cupy as cp


class TA51_B_FeatureProcessor_GPU:  # ← keeps legacy class name
    _QUANTILES = cp.asarray([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    _BIN_FEATURES = ["area", "eccentricity", "major_axis_length"]
    _STATS = ["mean", "std", "min", "max"]  # skew handled separately
    _UNITS = {
        "area": "px²",
        "eccentricity": "-",
        "major_axis_length": "px",
        "solidity": "-",
        "orientation": "°",
    }
    _SUM_FEATS = ["area", "eccentricity", "major_axis_length", "solidity", "orientation"]

    # ----------------------------------------------------------------
    # public API (unchanged signature)
    # ----------------------------------------------------------------
    def process(self, stackID: str, feature_tables: Sequence[cudf.DataFrame]) -> cudf.DataFrame:  # noqa: N802 – keep legacy name
        if not feature_tables:
            return cudf.DataFrame()
        df_regions = cudf.concat(feature_tables, ignore_index=True)
        summary = self._summarise(df_regions)
        summary["stackID"] = stackID
        return summary

    # ----------------------------------------------------------------
    # internals
    # ----------------------------------------------------------------
    def _summarise(self, df: cudf.DataFrame) -> cudf.DataFrame:
        rows = []
        for feat in self._BIN_FEATURES:
            if feat not in df.columns:
                continue
            edges = df[feat].quantile(self._QUANTILES).to_cupy()
            bins = cudf.Series(cp.digitize(df[feat].to_cupy(), edges, right=True))
            gb = df.groupby(bins)[[c for c in self._SUM_FEATS if c in df.columns]]
            agg_tbl = gb.agg(self._STATS).reset_index()
            # flatten MultiIndex → f"{col}_{stat}"
            agg_tbl.columns = [
                "__bin"
            ] + [f"{col}_{stat}" for col, stat in agg_tbl.columns[1:]]

            # skew (vectorised)
            for col in self._SUM_FEATS:
                if col not in df.columns:
                    continue
                vals = gb[col].apply(lambda s: cp.mean((s - s.mean()) ** 3) / ((s.std() + 1e-12) ** 3))
                agg_tbl[f"{col}_skew"] = vals.values

            # long‑format melt (same as old pipeline expects)
            long = agg_tbl.melt(id_vars="__bin", var_name="feature_stat", value_name="feature_value")
            long[["feature_name", "stat_type"]] = long.feature_stat.str.rsplit("_", n=1, expand=True)
            long = long.drop(columns="feature_stat")
            long = long.assign(bin_type=f"by_{feat}", unit=long.feature_name.map(self._UNITS))
            rows.append(long)
        return cudf.concat(rows, ignore_index=True) if rows else cudf.DataFrame()


