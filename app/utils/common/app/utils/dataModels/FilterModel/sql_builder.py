"""Translate a :class:`FilterModel` into a SQLAlchemy ``select``.

This layer depends **only** on SQLAlchemy and the FilterModel, keeping
ORM‑agnostic validation concerns decoupled from query‑generation.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Type

from sqlalchemy import select, and_, or_
from sqlalchemy.sql.elements import ColumnElement

from app.utils.common.app.utils.dataModels.FilterModel.FilterModel import FilterModel, Logic

__all__ = ["to_sqlalchemy"]

def _resolved_columns(orm_cls: Type, columns: Optional[Sequence[str]]):
    return [getattr(orm_cls, c) for c in columns] if columns else list(orm_cls.__table__.columns)


def to_sqlalchemy(*, orm_cls: Type, flt: FilterModel, columns: Optional[List[str]] = None):
    """Return a ready‑to‑execute SQLAlchemy ``Select`` object.

    Parameters
    ----------
    orm_cls : DeclarativeMeta
        SQLAlchemy ORM‑mapped class representing the target table.
    flt : FilterModel
        Validated filter description.
    columns : list[str] | None
        Columns to include; ``None`` ⇒ ``SELECT *``.
    """
    expr = _sql_expression(flt, orm_cls)
    return select(*_resolved_columns(orm_cls, columns)).where(expr)


def _sql_expression(flt: FilterModel, orm_cls: Type) -> ColumnElement:
    """
    Build a SQLAlchemy Boolean expression from the given FilterModel.

    Parameters
    ----------
    flt : FilterModel
        A validated FilterModel containing condition groups.
    orm_cls : type
        SQLAlchemy ORM-mapped class.

    Returns
    -------
    ColumnElement
        SQLAlchemy expression for use in WHERE clause.
    """
    exprs = [group.to_sql(orm_cls) for group in flt.groups]

    if not exprs:
        raise ValueError("No filter conditions provided.")

    return (
        and_(*exprs) if flt.global_logic == Logic.AND
        else or_(*exprs)
    )
