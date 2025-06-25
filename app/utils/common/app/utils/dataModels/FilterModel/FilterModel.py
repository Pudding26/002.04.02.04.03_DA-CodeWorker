from __future__ import annotations

from pydantic import BaseModel, Field
import pandas as pd
from typing import List, Optional
from pandas import Series, DataFrame
from app.utils.common.app.utils.dataModels.FilterModel.conditions import Condition, ConditionGroup, Logic, Op, Border


class FilterModel(BaseModel):
    """
    A structured, validated filter model representing a full filter query.

    Attributes:
        groups: List of condition groups (OR/AND within each).
        global_logic: Logical operator used to combine groups.
        job_id: Optional job metadata.
    """
    groups: List[ConditionGroup]
    global_logic: Logic = Logic.AND
    job_id: Optional[str] = Field(default=None)

    @classmethod
    def from_row(
        cls,
        row: Series,
        *,
        include_cols: list[str],
        range_cols: list[str] = None,
        min_cols: list[str] = None,
        max_cols: list[str] = None,
        border_rule: dict[str, Border] = None,
        global_logic: Logic = Logic.AND,
        job_id_field: Optional[str] = None,
    ) -> FilterModel:
        """
        Construct a FilterModel from a pandas Series (e.g. a DoE row).
        """
        range_cols = range_cols or []
        min_cols = min_cols or []
        max_cols = max_cols or []
        border_rule = border_rule or {}

        conds: list[Condition] = []
        for col in include_cols:
            val = row.get(col)

            # Skip if completely empty, string "None", or NaN
            if val is None or val == "None" or (not isinstance(val, (list, tuple)) and pd.isna(val)):
                continue

            # Always treat as list
            if not isinstance(val, (list, tuple)):
                val = [val]

            # Filter out junk values from the list
            val = [v for v in val if v is not None and v != "None" and not (isinstance(v, float) and pd.isna(v))]


            if not val:
                continue  # Nothing left to filter on

            if col in range_cols:
                ranges: list[tuple] = []
                for pair in val:
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        ranges.append(tuple(pair))
                if ranges:
                    conds.append(Condition(
                        column=col,
                        op=Op.BETWEEN,
                        value=ranges,  # 👈 List of ranges!
                        border=border_rule.get(col, Border.INCLUDE),
                    ))
            elif col in min_cols:
                conds.append(Condition(
                    column=col,
                    op=Op.GTE,
                    value=val[0],
                    border=border_rule.get(col, Border.INCLUDE),
                ))
            elif col in max_cols:
                conds.append(Condition(
                    column=col,
                    op=Op.LTE,
                    value=val[0],
                    border=border_rule.get(col, Border.INCLUDE),
                ))
            else:
                conds.append(Condition(
                    column=col,
                    op=Op.IN_,
                    value=val,
                ))

        group = ConditionGroup(logic=global_logic, conditions=conds)
        return cls(
            groups=[group],
            global_logic=Logic.AND,
            job_id=str(row.get(job_id_field)) if job_id_field else None
        )

    def apply_to_dataframe(self, df: DataFrame) -> DataFrame:
        """
        Apply the filter to a DataFrame in memory.
        """
        masks = [g.to_pd(df) for g in self.groups]
        result = masks[0]
        for m in masks[1:]:
            result = result & m if self.global_logic == Logic.AND else result | m
        return df[result].copy()

    def has_conditions(self) -> bool:
        """Check whether any group contains conditions."""
        return any(g.conditions for g in self.groups)
    
    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        *,
        include_cols: list[str],
        is_range_cols: list[str] = None,
        is_max: list[str] = None,
        is_min: list[str] = None,
        border_rule: dict[str, Border] = None,
        global_logic: Logic = Logic.AND,
        job_id_field: str = "job_uuid"
    ) -> list["FilterModel"]:
        """
        Convert an entire DataFrame of job specs into a list of FilterModels.

        Parameters:
            df: DataFrame containing filter fields
            include_cols: Columns to include as filters
            is_range_cols: Columns treated as BETWEEN
            is_max: Columns treated as <=
            is_min: Columns treated as >=
            border_rule: Dict of borders (INCLUDE/EXCLUDE) for range filters
            global_logic: AND/OR logic for combining filter groups
            job_id_field: Column name to assign as job_id in FilterModel

        Returns:
            List of FilterModel instances, one per row
        """
        return [
            FilterModel.from_row(
                row=row,
                include_cols=include_cols,
                range_cols=is_range_cols or [],
                min_cols=is_min or [],
                max_cols=is_max or [],
                border_rule={k: Border(border_rule[k]) for k in border_rule} if border_rule else {},
                global_logic=global_logic,
                job_id_field=job_id_field
            )
            for _, row in df.iterrows()
        ]


    @staticmethod
    def from_human_filter(human_filter: dict) -> FilterModel:
        """
        Construct a FilterModel from a human-friendly filter dictionary.
        """
        contains = human_filter.get("contains", {})
        notcontains = human_filter.get("notcontains", {})
        is_range = set(human_filter.get("is_range", []))
        global_logic = Logic.OR if human_filter.get("all_OR") else Logic.AND

        def parse_conditions(section: dict, negated: bool = False) -> list[Condition]:
            conds = []
            for col, val in section.items():
                border = Border.INCLUDE if col in is_range else None

                # Normalize val to expected format
                if isinstance(val, str) or not isinstance(val, (list, dict)):
                    val = [val]

                if isinstance(val, dict):
                    if "or" in val:
                        for or_item in val["or"]:
                            if isinstance(or_item, list) and len(or_item) == 2 and col in is_range:
                                conds.append(Condition(
                                    column=col,
                                    op=Op.NOT_BETWEEN if negated else Op.BETWEEN,
                                    value=[tuple(or_item)],
                                    border=border
                                ))
                            else:
                                conds.append(Condition(
                                    column=col,
                                    op=Op.NOT_IN if negated else Op.IN_,
                                    value=or_item
                                ))
                    elif "and" in val:
                        conds.extend([
                            Condition(
                                column=col,
                                op=Op.NOT_IN if negated else Op.IN_,
                                value=[v]
                            ) for v in val["and"]
                        ])
                else:
                    # Simple value list
                    for v in val:
                        conds.append(Condition(
                            column=col,
                            op=Op.NOT_IN if negated else Op.IN_,
                            value=[v]
                        ))

            return conds

        conditions = parse_conditions(contains, negated=False) + parse_conditions(notcontains, negated=True)
        group = ConditionGroup(logic=global_logic, conditions=conditions)
        return FilterModel(groups=[group], global_logic=Logic.AND)


