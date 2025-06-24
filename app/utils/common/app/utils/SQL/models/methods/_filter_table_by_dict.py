from sqlalchemy.orm import Session
from sqlalchemy import and_, not_
from typing import Type, Dict, List, Literal, Any
import logging


def _filter_table_by_dict(
    orm_class: Type,
    session: Session,
    filters: Dict[str, List[Any]],
    method: Literal["keep", "drop"] = "drop"
) -> Dict[str, Any]:
    """
    Filter rows directly in the DB table based on filter conditions.

    Returns:
        dict: Summary with counts.
    """
    if method not in {"keep", "drop"}:
        raise ValueError("method must be 'keep' or 'drop'")
    if not filters:
        raise ValueError("filters cannot be empty")


    before_count = session.query(orm_class).count()

    # Track per-column matching stats
    per_column_match_count = {}
    for col_name, values in filters.items():
        col = getattr(orm_class, col_name, None)
        if col is None:
            raise ValueError(f"Column '{col_name}' not found in ORM class {orm_class.__name__}")
        match_count = session.query(orm_class).filter(col.in_(values)).count()
        per_column_match_count[col_name] = match_count

    # Build compound condition
    conditions = [
        getattr(orm_class, col_name).in_(values)
        for col_name, values in filters.items()
    ]
    final_condition = and_(*conditions)
    delete_condition = final_condition if method == "drop" else not_(final_condition)

    # Execute DELETE
    deleted_count = session.query(orm_class).filter(delete_condition).delete(synchronize_session=False)
    session.commit()

    after_count = session.query(orm_class).count()

    # 🔍 Formatted log summary
    logging.debug3(
        f"\n🧹 Table Filtering Summary ({method.upper()}):\n"
        f"   • Rows before: {before_count:,}\n"
        f"   • Rows after:  {after_count:,}\n"
        f"   • Rows deleted: {deleted_count:,}\n"
        f"   • Filter impact by column:"
    )
    for col, count in per_column_match_count.items():
        logging.debug2(f"\n     → '{col}' matched {count:,} row(s)")


