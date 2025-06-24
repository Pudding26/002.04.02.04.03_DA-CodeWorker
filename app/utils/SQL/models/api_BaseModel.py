import logging
from time import time
import pandas as pd
import numpy as np


from contextvars import ContextVar
from sqlalchemy import select
from uuid import uuid4
from tqdm import tqdm

from pydantic import BaseModel
from app.utils.SQL.models import enums

from app.utils.SQL.errors import BulkInsertError


from app.utils.SQL.models.methods._bulk_save_objects import _bulk_save_objects
from app.utils.SQL.models.methods._model_validate_dataframe import _model_validate_dataframe
from app.utils.SQL.models.methods._filter_table_by_dict import _filter_table_by_dict

from app.utils.dataModels.FilterModel.FilterModel import FilterModel


from app.utils.SQL.SQL_FetchBuilder import SQL_FetchBuilder
from app.utils.SQL.to_SQLSanitizer import to_SQLSanitizer
from app.utils.QM.PydanticQM import PydanticQM


from typing import List, Type, TypeVar, Optional, Any, Dict, ClassVar, Union, get_origin



from app.utils.SQL.DBEngine import DBEngine



_cid: ContextVar[str] = ContextVar("_cid")

T = TypeVar("T", bound="SharedBaseModel")

class api_BaseModel(BaseModel):
    """
    Shared base for all Pydantic models in the application.
    """
    Enums: ClassVar = enums # Make the enums available to all models
    class Config:
        from_attributes = True
        str_strip_whitespace = True
        anystr_strip_whitespace = True
        use_enum_values = True
        extra = "forbid"

    def to_dict(self, exclude_none: bool = True) -> dict:
        """Convert to dict."""
        return self.model_dump(exclude_none=exclude_none)

    def json_api(self, **kwargs: Any) -> str:
        """Pretty JSON for APIs/logs/debugging."""
        return self.model_dump_json(indent=2, exclude_none=True, **kwargs)
    

    @classmethod
    def store_dataframe(
        cls,
        df: pd.DataFrame,
        db_key: str = None,
        method: str = "append",
        insert_method: str = "bulk_save_objects",
    ) -> None:
        """
        • Cleans & validates a DataFrame with the model schema.
        • Persists it using one of three strategies:
            - bulk_save_objects   (default, debug-aware)
            - bulk_insert_mappings
            - to_sql
        • If `method == "replace"` the target table is truncated first.
        """
        from app.utils.SQL.DBEngine import DBEngine
        from sqlalchemy.orm import sessionmaker
        
        
        if db_key is not None:
            logging.warning(f"🔍 DEPRECATED; db_key is fetched from api_instance.")

        db_key = cls.db_key



        engine  = DBEngine(db_key).get_engine()
        Session = sessionmaker(bind=engine)
        session = Session()
        table_name = cls.orm_class.__tablename__

        try:
            def _log_step(df, step_name, steps_log):
                shape = df.shape
                steps_log.append(f"{step_name}: shape = {shape}")
                return df
            # 1.  sanitise → coerce → drop incomplete
            
            steps_log = []
            df = df.copy()
            df = _log_step(to_SQLSanitizer().sanitize(df), "sanitize", steps_log)
            df = _log_step(to_SQLSanitizer.sanitize_columns_from_model(df, cls), "sanitize_columns_from_model", steps_log)
            df = _log_step(to_SQLSanitizer.coerce_numeric_fields_from_model(df, cls), "coerce_numeric_fields_from_model", steps_log)
            df = _log_step(to_SQLSanitizer.coerce_string_fields_from_model(df, cls), "coerce_string_fields_from_model", steps_log)
            df = _log_step(to_SQLSanitizer.coerce_datetime_fields_from_model(df, cls), "coerce_datetime_fields_from_model", steps_log)
            df = _log_step(to_SQLSanitizer.drop_incomplete_rows_from_model(df, cls), "drop_incomplete_rows_from_model", steps_log)
            df = _log_step(to_SQLSanitizer.drop_invalid_enum_rows_from_model(df, cls), "drop_invalid_enum_rows_from_model", steps_log)
            df = _log_step(to_SQLSanitizer().sanitize(df), "final sanitize", steps_log)

            summary = "\n".join(steps_log)
            logging.debug3(f"\n=== DataFrame Shape Report ===\n{summary}")




            # 2.  validate with Pydantic
            validated = _model_validate_dataframe(cls, df)



            # 3.  optional table truncate
            if method == "replace":
                logging.warning(f"⚠️ 'replace' deletes all rows in {cls.orm_class.__tablename__}")
                session.query(cls.orm_class).delete()
                session.commit()


            # 4.  choose persistence strategy
            start = time()
            if insert_method == "bulk_save_objects":
                _bulk_save_objects(
                    orm_cls=cls.orm_class,
                    records=validated,
                    session=session,
                    batch_size=5_000,
                    switch_threshold=5,
                    abort_threshold=15,
                )
                session.commit()

            elif insert_method == "bulk_insert_mappings":
                session.bulk_insert_mappings(
                    cls.orm_class,
                    validated.to_dict(orient="records"),
                )
                session.commit()

            elif insert_method == "to_sql":
                validated.to_sql(
                    name=cls.orm_class.__tablename__,
                    con=engine,
                    if_exists=method,       # 'append' or 'replace'
                    index=False,
                    method="multi",
                    chunksize=5_000,
                )

            else:
                raise ValueError(f"Unknown insert_method: {insert_method}")

            logging.info(
                f"✅ Stored {len(validated)} rows to "
                f"{cls.orm_class.__tablename__} in {time()-start:.2f}s "
                f"using {insert_method}"
            )

        except BulkInsertError:
            # already logged in detail by the helper – don't spam again
            session.rollback()
            raise

        except Exception as exc:
            session.rollback()
            logging.error(f"❌ store_dataframe failed: {exc}", exc_info=True)
        
        finally:
            session.close()


    @classmethod
    def filter_table_by_dict(
        cls,
        filter_dict: Dict[str, List[Any]],
        method: str = "drop",
        db_key: Optional[str] = None,
        orm_class: Optional[Type] = None
    ) -> int:
        """
        Delete rows in the table based on matching filter criteria.
        
        Example:
            MyModel._filter_table_by_dict({"status": ["inactive"]}, method="drop")
        
        Returns:
            int: Number of rows deleted.
        """
        from app.utils.SQL.DBEngine import DBEngine

        orm_class = orm_class or getattr(cls, "orm_class", None)
        db_key = db_key or getattr(cls, "db_key", "raw")

        if orm_class is None:
            raise ValueError("Missing orm_class")

        session = DBEngine(db_key).get_session()

        try:
            return _filter_table_by_dict(
                orm_class=orm_class,
                session=session,
                filters=filter_dict,
                method=method
            )
        finally:
            session.close()




    @classmethod
    def fetch(
        cls,
        method: str = None,
        filter_model: Optional[FilterModel] = None,
        filter_dict: Optional[Dict[str, List[Any]]] = None,
        db_key: Optional[str] = None,
        orm_class: Optional[type] = None,
        columns: Optional[list[str]] = None,
        stream: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch data from the database with optional filtering and column projection.
        Supports FilterModel-based structured filtering or legacy filter_dict.
        """
    
        if method is not None:
            logging.warning(f"🔍 DEPRECATED; ALL is default and dsticntion comes form provided Filter or not")
        if filter_dict is not None:
            logging.warning(f"🔍 DEPRECATED; use FilterModel instead of filter_dict")
        
        cid = uuid4().hex[:8]
        _cid.set(cid)
        ctx = {"cid": cid}

        orm_class = orm_class or getattr(cls, "orm_class", None)
        db_key = db_key or getattr(cls, "db_key", "raw")
        if orm_class is None:
            raise ValueError("orm_class missing")

        logging.info("📥 fetch start", extra=ctx)
        start = time()

        session = DBEngine(db_key).get_session()
        try:
            # 🔧 Unified fetch builder: uses filter_model if provided
            builder = SQL_FetchBuilder(orm_class, filter_model)
            stmt = builder.build_select(method, columns).execution_options(stream_results=True)

            logging.debug1("📝 SQL built", extra=ctx)
            logging.debug1(
                stmt.compile(compile_kwargs={"literal_binds": True}),
                extra=ctx,
            )

            # 📤 Execute and collect
            if stream:
                result = session.execute(stmt)
                rowcount = getattr(builder, "rowcount", None)
                data = [
                    cls.model_validate(r._mapping).model_dump()
                    for r in tqdm(result.yield_per(5000), desc="stream", total=rowcount)
                ]
            else:
                data = [
                    cls.model_validate(r._mapping).model_dump()
                    for r in session.execute(stmt).all()
                ]

            df = pd.DataFrame(data, columns=columns or list(cls.model_fields))
            logging.info(f"✅ fetched {len(df)} rows in {time()-start:.2f}s", extra=ctx)
            logging.debug2(f"🔑 digest={df.select_dtypes('number').sum().sum()}", extra=ctx)
            return df

        except Exception as exc:
            logging.error(f"❌ fetch failed: {exc}", extra=ctx, exc_info=True)
            return pd.DataFrame()
        finally:
            session.close()


    @classmethod
    def fetch_distinct_values(
        cls,
        column: str,
        db_key: Optional[str] = None,
        orm_class: Optional[type] = None
    ) -> list:
        """
        Efficiently fetch distinct values of a single column from the DB.
        """
        from app.utils.SQL.models.methods._fetch_distinct_values import _fetch_distinct_values
        return _fetch_distinct_values(
            cls=cls,
            column=column,
            db_key=db_key,
            orm_class=orm_class
        )



    @classmethod
    def validate_df(cls, df: pd.DataFrame) -> List[dict]:
        """
        Validate DataFrame rows using Pydantic. Returns a list of validation errors, if any.
        Each error is a dict: {'row': index, 'error': str, 'row_data': dict}
        """
        errors = []
        records = df.to_dict(orient="records")

        for idx, record in enumerate(records):
            try:
                cls.model_validate(record)
            except Exception as e:
                errors.append({
                    "row": idx,
                    "error": str(e),
                    "row_data": record
                })
        return errors


    @classmethod
    def _db_shape(cls, db_key: str = None, orm_class: Optional[Type] = None) -> tuple[int, int]:
        """
        Return the shape (rows, columns) of the SQL table mapped to this model.
        """
        orm_class = orm_class or getattr(cls, "orm_class", None)
        db_key = db_key or getattr(cls, "db_key", "raw")

        if orm_class is None:
            raise ValueError(f"{cls.__name__} must define 'orm_class' or pass it explicitly.")

        session = DBEngine(db_key).get_session()

        try:
            row_count = session.query(orm_class).count()
            column_count = len(cls.model_fields)
            return (row_count, column_count)
        except Exception as e:
            logging.error(f"❌ Failed to get DB shape for {orm_class.__tablename__}: {e}", exc_info=True)
            return (0, 0)
        finally:
            session.close()


    @staticmethod
    def report_fake_Nulls(df: pd.DataFrame) -> pd.DataFrame:
        return to_SQLSanitizer().detect_fakes(df)


    @classmethod
    def update_row(cls, row_data: dict, match_cols: list[str] = ["job_uuid"], db_key: Optional[str] = None) -> None:
        """
        Generic update method: match on `match_cols`, update all values in `row_data`.
        Assumes row exists.
        """
        from sqlalchemy.orm import Session
        from app.utils.SQL.DBEngine import DBEngine

        db_key = db_key or getattr(cls, "db_key", "raw")
        orm_class = getattr(cls, "orm_class", None)
        if orm_class is None:
            raise ValueError(f"{cls.__name__} must define orm_class.")

        session: Session = DBEngine(db_key).get_session()
        try:
            filters = {col: row_data[col] for col in match_cols}
            row = session.query(orm_class).filter_by(**filters).first()

            if not row:
                raise ValueError(f"No matching row found using {filters}")

            for k, v in row_data.items():
                if hasattr(row, k):
                    setattr(row, k, v)

            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()



    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame, groupby_col: Any = None) -> pd.DataFrame:
        return PydanticQM.evaluate(df, groupby_col=groupby_col)

    @classmethod
    def prepare_dataframe(cls, df: pd.DataFrame, instructions: Dict[str, Any]) -> pd.DataFrame:
        return PydanticQM.clean_and_coerce(df, model=cls, instructions=instructions)
    
    @classmethod
    def plot_report(cls, df_report: pd.DataFrame, top_n: int = 10, grouped: bool = None) -> List[str]:
        return PydanticQM.plot_report(df_report=df_report,top_n=top_n, grouped =grouped)

    @classmethod
    def make_sql_safe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert DataFrame to SQL-safe format:
        - Convert all columns to object type
        - Replace NaN with None
        """
        return df.astype(object).where(pd.notnull(df), None)

    @classmethod
    def pydantic_model_to_dtype_dict(cls: type[BaseModel]) -> dict[str, type]:
        dtype_map = {}
        for name, field in cls.model_fields.items():
            outer_type = field.annotation
            origin = get_origin(outer_type) or outer_type

            # Handle common typing cases
            if origin in (list, dict):
                dtype_map[name] = object
            elif origin is float:
                dtype_map[name] = float
            elif origin is int:
                dtype_map[name] = int
            elif origin is bool:
                dtype_map[name] = bool
            elif origin is str:
                dtype_map[name] = str
            else:
                dtype_map[name] = object  # fallback

        return dtype_map

####### DEPRECATED METHODS ########

    @classmethod
    def db_shape(cls, db_key: str = None, orm_class: Optional[Type] = None) -> tuple[int, int]:
        """
        Return the shape (rows, columns) of the SQL table mapped to this model.
        If the query fails, return (0, 0).
        """
        logging.warning(f"🔍 DEPRECATED use _db_shape instead. => redirected to _db_shape")
        return cls._db_shape(db_key=db_key, orm_class=orm_class)



    @classmethod
    def fetch_all(cls: Type[T], db_key: str = None, orm_class: Optional[Type] = None) -> pd.DataFrame:
        """
        Fetch all entries from the database and return as a DataFrame.
        If the query succeeds but returns no rows, return a DataFrame with correct schema.
        If the query fails, return a truly empty DataFrame.
        """
        logging.warning(f"🔍 DEPRECATED use fetch instead. => redirected to fetch with method = all")
        return cls.fetch(db_key=db_key, orm_class=orm_class, method="all")

        orm_class = orm_class or getattr(cls, "orm_class", None)
        db_key = db_key or getattr(cls, "db_key", "raw")

        if orm_class is None:
            raise ValueError(f"{cls.__name__} must define 'orm_class' or pass it explicitly.")
        
        session = DBEngine(db_key).get_session()
        logging.debug2(f"🔍 Fetching all entries from {orm_class.__name__} in {db_key} database")

        try:
            results = session.query(orm_class).all()

            if not results:
                logging.debug1(f"📭 No entries found in table {orm_class.__tablename__}")
                field_names = list(cls.model_fields)
                return pd.DataFrame(columns=field_names)

            validated = [cls.model_validate(row).model_dump() for row in results]
            return pd.DataFrame(validated)

        except Exception as e:
            logging.error(f"❌ Failed to fetch {orm_class.__name__} entries: {e}", exc_info=True)
            return pd.DataFrame()
        finally:
            session.close()