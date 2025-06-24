from sqlalchemy import select, and_, or_, not_, between
from sqlalchemy.orm.attributes import InstrumentedAttribute
from typing import Type, Optional


from app.utils.dataModels.FilterModel.FilterModel import FilterModel
from app.utils.dataModels.FilterModel.sql_builder import to_sqlalchemy

class SQL_FetchBuilder:
    """
    Builds SQLAlchemy Select objects for fetching model data.
    Supports complex filtering logic: AND/OR per-column, global AND/OR flags,
    and BETWEEN-range support via `is_range`.
    """

    def __init__(
        self,
        orm_class: Type,
        filter_model: Optional[FilterModel] = None
    ):
        self.orm_class = orm_class
        self.filter_model = filter_model


    def build_select(self, method: str, columns: Optional[list[str]] = None):
        
        if self.filter_model:
            return to_sqlalchemy(orm_cls=self.orm_class, flt=self.filter_model, columns=columns)
        
        else:
            return self._build_all_select(columns)



    def _build_all_select(self, columns: Optional[list[str]] = None):
        selected = self._resolve_columns(columns)
        return select(*selected)

    def _resolve_columns(self, columns: Optional[list[str]]):
        if columns:
            try:
                return [getattr(self.orm_class, col) for col in columns]
            except AttributeError as e:
                raise ValueError(f"Invalid column name in projection: {e}")
        else:
            return list(self.orm_class.__table__.columns)
