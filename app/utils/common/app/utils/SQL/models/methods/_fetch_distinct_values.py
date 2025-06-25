from sqlalchemy import select, distinct
from typing import Optional, Type

from app.utils.common.app.utils.SQL.DBEngine import DBEngine


def _fetch_distinct_values(
    cls,
    column: str,
    db_key: Optional[str] = None,
    orm_class: Optional[Type] = None
) -> list:
    """
    Efficiently fetch distinct values of a single column from the DB.
    """
    orm_class = orm_class or getattr(cls, "orm_class", None)
    db_key = db_key or getattr(cls, "db_key", "raw")

    if orm_class is None:
        raise ValueError("Missing orm_class")

    session = DBEngine(db_key).get_session()
    try:
        stmt = select(distinct(getattr(orm_class, column)))
        result = session.execute(stmt).scalars().all()
        return result
    finally:
        session.close()
