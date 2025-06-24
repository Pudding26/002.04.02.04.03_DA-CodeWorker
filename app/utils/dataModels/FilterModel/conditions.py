from enum import Enum
from typing import List, Union, Tuple, Any
from pydantic import BaseModel
from sqlalchemy import and_, or_, not_, between
from sqlalchemy.sql.elements import ColumnElement
import pandas as pd


class Logic(str, Enum):
    AND = "and"
    OR = "or"


class Op(str, Enum):
    IN_ = "in"
    NOT_IN = "not in"
    BETWEEN = "between"
    GTE = ">="
    LTE = "<="
    GT = ">"
    LT = "<"


class Border(str, Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"


class Condition(BaseModel):
    column: str
    op: Op
    value: Union[Any, List[Any], Tuple[Any, Any]]
    border: Border = Border.INCLUDE

    def to_sql(self, orm_cls: type) -> ColumnElement:
        col = getattr(orm_cls, self.column)

        match self.op:
            case Op.IN_:
                return col.in_(self.value)
            case Op.NOT_IN:
                return not_(col.in_(self.value))
            case Op.BETWEEN:
                if isinstance(self.value, list):
                    return or_(
                        between(col, v[0], v[1]) for v in self.value
                    )
                elif isinstance(self.value, tuple):
                    return between(col, self.value[0], self.value[1])
                else:
                    raise ValueError("BETWEEN filter expects a tuple or list of tuples")
            case Op.GTE:
                return col >= self.value if self.border == Border.INCLUDE else col > self.value
            case Op.LTE:
                return col <= self.value if self.border == Border.INCLUDE else col < self.value
            case Op.GT:
                return col > self.value
            case Op.LT:
                return col < self.value
            case _:
                raise ValueError(f"Unsupported operation: {self.op}")

    def to_pd(self, df: pd.DataFrame) -> pd.Series:
        col = df[self.column]

        match self.op:
            case Op.IN_:
                return col.isin(self.value)
            case Op.NOT_IN:
                return ~col.isin(self.value)
            case Op.BETWEEN:
                low, high = self.value
                if self.border == Border.INCLUDE:
                    return col.between(low, high)
                return (col > low) & (col < high)
            case Op.GTE:
                return col >= self.value if self.border == Border.INCLUDE else col > self.value
            case Op.LTE:
                return col <= self.value if self.border == Border.INCLUDE else col < self.value
            case Op.GT:
                return col > self.value
            case Op.LT:
                return col < self.value
            case _:
                raise ValueError(f"Unsupported operation: {self.op}")


class ConditionGroup(BaseModel):
    conditions: List[Condition]
    logic: Logic = Logic.AND

    def to_sql(self, orm_cls: type) -> ColumnElement:
        exprs = [cond.to_sql(orm_cls) for cond in self.conditions]
        return and_(*exprs) if self.logic == Logic.AND else or_(*exprs)

    def to_pd(self, df: pd.DataFrame) -> pd.Series:
        masks = [cond.to_pd(df) for cond in self.conditions]
        if not masks:
            return pd.Series([True] * len(df))
        result = masks[0]
        for mask in masks[1:]:
            result = result & mask if self.logic == Logic.AND else result | mask
        return result
