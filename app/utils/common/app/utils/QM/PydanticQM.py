import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Type, Any, Dict
import logging



class PydanticQM:

    @staticmethod
    def clean_and_coerce(df: pd.DataFrame, model: Type, instructions: Dict[str, Any]) -> pd.DataFrame:
        df = df.copy()
        tasks = instructions.get("tasks", [])
        rename_map = instructions.get("colNameMapper", {})

        if "rename" in tasks and rename_map:
            df.rename(columns=rename_map, inplace=True)

        df.replace(["NaN", "nan", "", pd.NA, np.nan], value=None, inplace=True)
        df = df.astype(object)

        if "dtypes" in tasks:
            for field, field_info in model.model_fields.items():
                if field not in df.columns:
                    df[field] = None
                    continue

                target_type = field_info.annotation
                try:
                    if target_type is datetime:
                        df[field] = pd.to_datetime(df[field], errors="coerce")
                    elif target_type is bool:
                        df[field] = df[field].map(lambda x: None if x is None else bool(x))
                    elif target_type in [int, float, str]:
                        df[field] = df[field].map(lambda x: None if x is None else target_type(x))
                except Exception as e:
                    print(f"⚠️ Failed to cast column '{field}' to {target_type}: {e}")

        return df

    @staticmethod
    def evaluate(df: pd.DataFrame, groupby_col: Any = None) -> pd.DataFrame:
        df = df.copy()

        def _evaluate_single(sub_df: pd.DataFrame) -> pd.DataFrame:
            report = {}
            for col in sub_df.columns:
                data = sub_df[col]
                null_count = data.isna().sum()
                total = len(data)
                sample_types = data.dropna().map(type).value_counts().to_dict()
                report[col] = {
                    "nulls": null_count,
                    "non_nulls": total - null_count,
                    "null_%": round(null_count / total * 100, 2) if total else 0.0,
                    "sample_types": sample_types
                }
            return pd.DataFrame(report).T

        if groupby_col:
            grouped = df.groupby(groupby_col)
            all_reports = []
            for group_val, sub_df in grouped:
                report = _evaluate_single(sub_df)
                report["group"] = group_val
                report.index.name = "column"
                all_reports.append(report.reset_index())

            final = pd.concat(all_reports).set_index(["group", "column"])
            return final.sort_index()

        return _evaluate_single(df).sort_values("null_%", ascending=False)





    @staticmethod
    def plot_report(df_report: pd.DataFrame, top_n: int = 15, grouped: bool = None) -> list:
        """
        Generate and save visual QA report.

        Parameters:
        - df_report: output from evaluate()
        - top_n: number of columns to include in plots
        - grouped: override auto-detection of grouped report

        Returns:
        - List of file paths to saved plots
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os, glob

        plot_dir = "temp/reports/df_qa"
        os.makedirs(plot_dir, exist_ok=True)

        def _save_plot(fig, filename):
            existing = sorted(glob.glob(os.path.join(plot_dir, "*.png")), key=os.path.getmtime)
            while len(existing) >= 10:
                os.remove(existing.pop(0))
            path = os.path.join(plot_dir, filename)
            fig.savefig(path)
            plt.close(fig)
            return path

        plot_paths = []
        if grouped is None:
            grouped = isinstance(df_report.index, pd.MultiIndex)

        if grouped:
            # Barplot: total nulls per column
            summary = df_report.groupby("column")["nulls"].sum().sort_values(ascending=False).head(top_n)
            fig, ax = plt.subplots(figsize=(10, 6))
            summary.plot(kind="barh", ax=ax)
            ax.set_title("Top Columns with Most Nulls Across Groups")
            ax.set_xlabel("Total Nulls")
            plot_paths.append(_save_plot(fig, "grouped_total_nulls.png"))

            # Heatmap
            pivot = df_report.reset_index().pivot(index="group", columns="column", values="null_%")
            fig, ax = plt.subplots(figsize=(12, 6))
            cax = ax.imshow(pivot.fillna(0), cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=90)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            fig.colorbar(cax, ax=ax, label="Null %")
            ax.set_title("Null Percentage by Group and Column")
            plot_paths.append(_save_plot(fig, "grouped_null_heatmap.png"))

            # Hue barplot
            df_plot = df_report.reset_index()
            top_cols = (
                df_plot.groupby("column")["nulls"]
                .sum()
                .sort_values(ascending=False)
                .head(top_n)
                .index.tolist()
            )
            filtered = df_plot[df_plot["column"].isin(top_cols)]
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(
                data=filtered, y="column", x="nulls", hue="group",
                estimator=sum, dodge=True, ax=ax
            )
            ax.set_title("Top Columns with Most Nulls by Group (Hue)")
            ax.set_xlabel("Null Count")
            ax.set_ylabel("Column")
            plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plot_paths.append(_save_plot(fig, "grouped_nulls_hue.png"))

        else:
            # Null % by column
            fig, ax = plt.subplots(figsize=(10, 6))
            df_report["null_%"].sort_values(ascending=False).head(top_n).plot(
                kind="barh", ax=ax
            )
            ax.set_title("Top Columns by Null Percentage")
            ax.set_xlabel("Null %")
            plot_paths.append(_save_plot(fig, "global_null_percent.png"))

            # Type diversity
            diversity = df_report["sample_types"].map(lambda d: len(d) if isinstance(d, dict) else 0)
            fig, ax = plt.subplots(figsize=(10, 6))
            diversity.sort_values(ascending=False).head(top_n).plot(
                kind="barh", ax=ax
            )
            ax.set_title("Columns with Most Data Type Variability")
            ax.set_xlabel("Distinct Python Types")
            plot_paths.append(_save_plot(fig, "global_dtype_diversity.png"))

        return plot_paths

