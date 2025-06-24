from __future__ import annotations


import numpy as np
import pandas as pd
from collections import defaultdict
from enum import Enum
from typing import List, Union, get_args, get_origin, Optional, Type
from pydantic import BaseModel
import logging


class to_SQLSanitizer():
    def __init__(self, fake_nulls: Union[List[Union[str, float]], None] = None):
        self.fake_nulls = fake_nulls or [
            "", " ", "NaN", "nan", "NAN", "null", "None", "none",
            "missing", "-", "_", np.nan, pd.NA, pd.NaT
        ]

    def to_object(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.astype(object)

    def detect_fakes(self, df: pd.DataFrame) -> pd.DataFrame:
        report = defaultdict(dict)
        for col in df.columns:
            series = df[col]
            for fake in self.fake_nulls:
                if pd.isna(fake):
                    count = series.isna().sum()
                    key = "np.nan"
                else:
                    count = (series == fake).sum()
                    key = str(fake)
                if count > 0:
                    report[col][key] = int(count)
        return pd.DataFrame(report).T.fillna(0).astype(int)

    def replace_fakes(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.debug2("🔍 SQLSanitizer: checking for fake nulls...")
        df_clean = df.copy()
        report = self.detect_fakes(df_clean)

        if report.empty:
            logging.debug2("✅ SQLSanitizer finished: no fake nulls found.")
            return df_clean

        total_fixes = int(report.sum().sum())
        per_column_fixes = report.sum(axis=1).astype(int).to_dict()
        col_fix_summary = ', '.join([f"{col}: {cnt}" for col, cnt in per_column_fixes.items()])

        df_clean.replace(to_replace=self.fake_nulls, value=None, inplace=True)
        df_clean = df_clean.where(pd.notnull(df_clean), None)

        logging.debug2(f"✅ SQLSanitizer finished: replaced {total_fixes} fake nulls. "
                       f"Per-column summary → {col_fix_summary}")
        return df_clean

    def to_python_scalars(self, df: pd.DataFrame, log: bool = True) -> pd.DataFrame:
        df_clean = df.copy()
        conversions_per_col = {}

        def convert_and_count(series):
            count = 0
            def safe_cast(val):
                nonlocal count
                if hasattr(val, "item"):
                    count += 1
                    return val.item()
                return val
            new_series = series.map(safe_cast)
            conversions_per_col[series.name] = count
            return new_series

        df_clean = df_clean.apply(convert_and_count)

        if log:
            affected = {k: v for k, v in conversions_per_col.items() if v > 0}
            if affected:
                summary = ", ".join(f"{k}: {v}" for k, v in affected.items())
                logging.debug2(f"📤 SQLSanitizer scalar cast: converted values per column → {summary}")
            else:
                logging.debug2("📤 SQLSanitizer scalar cast: no NumPy scalars found.")

        return df_clean

    @staticmethod
    def coerce_numeric_fields_from_model(df: pd.DataFrame, model_cls: Type) -> pd.DataFrame:
        """
        Coerce every column whose type annotation in `model_cls` is `int`/`float`
        (or `Optional[int]`, `Optional[float]`) to the corresponding **nullable**
        pandas dtype (Int64 / Float64). After coercion any <NA> values are
        converted to real Python ``None`` so that Pydantic can handle them
        correctly.

        A concise debug report is emitted via ``logging.debug2`` / ``logging.debug3``.
        """
        df = df.copy()
        logging.debug3("🔤 Coercing numeric fields from model...")

        coerced_columns: dict[str, int] = {}
        nullified_columns: dict[str, int] = {}

        # ------------------------------------------------------------------ helpers
        def _is_numeric(typ, kind):
            origin = get_origin(typ)
            args   = get_args(typ)
            if kind is int:
                return typ is int or (origin in (Optional, Union) and int in args)
            if kind is float:
                return typ is float or (origin in (Optional, Union) and float in args)
            return False

        failed_columns: dict[str, str] = {}

        # ------------------------------------------------------------------ main loop
        for field_name, field in model_cls.model_fields.items():
            if field_name not in df.columns:
                logging.warning(f"❌ Column '{field_name}' not found in DataFrame, skipping coercion.")
                continue

            typ = field.annotation
            if _is_numeric(typ, int):
                target_dtype = "Int64"  # pandas *nullable* int
            elif _is_numeric(typ, float):
                target_dtype = "Float64"  # pandas *nullable* float
            else:
                continue  # → not numeric

            original = df[field_name]

            try:
                coerced_numeric = pd.to_numeric(original, errors="coerce")

                if target_dtype == "Int64":
                    non_na = coerced_numeric.dropna()
                    if not (non_na % 1 == 0).all():
                        failed_columns[field_name] = "non-integer float values"
                        logging.warning(f"❌ Column '{field_name}': contains non-integer floats, cannot coerce to Int64")
                        continue

                coerced = coerced_numeric.astype(target_dtype)

            except Exception as e:
                failed_columns[field_name] = str(e)
                logging.error(f"❌ Column '{field_name}' failed coercion to {target_dtype}: {e}")
                continue

            changed_mask = (original != coerced) & ~(original.isna() & coerced.isna())
            coerced_columns[field_name] = int(changed_mask.sum())
            if changed_mask.any():
                examples = original[changed_mask].dropna().astype(str).unique()[:3]
                logging.debug1(f"🔄 Column '{field_name}': coerced {changed_mask.sum()} "
                               f"values to {target_dtype}. Examples: {list(examples)}")
            else:
                logging.debug1(f"✅ Column '{field_name}': no coercion needed")

            # Replace <NA> with None
            coerced = coerced.astype(object)
            coerced = coerced.where(pd.notnull(coerced), None)
            na_after = sum(val is None for val in coerced)
            if na_after:
                nullified_columns[field_name] = na_after
                logging.debug1(f"⚠️ Column '{field_name}': {na_after} <NA> replaced with None")

            df[field_name] = coerced

        # ------------------------------------------------------------------ failure summary
        if failed_columns:
            logging.debug2(
                "\n⛔ Numeric Coercion Failures:\n" +
                "\n".join(f"   - {col}: {reason}" for col, reason in failed_columns.items())
            )


        total_coerced = sum(coerced_columns.values())
        total_nullified = sum(nullified_columns.values())


        # ------------------------------------------------------------------ summary
        logging.debug3(f"\n📋Numeric Coercion Complete: {total_coerced} coerced, {total_nullified} nullified")
        if coerced_columns:
            summary = ', '.join(f"{k}={v}" for k, v in coerced_columns.items())
            logging.debug2(f"🔢 Numeric Coercion: {summary}")
        if nullified_columns:
            summary = ', '.join(f"{k}={v}" for k, v in nullified_columns.items())
            logging.debug2(f"⚠️ Numeric Nulls: {summary}")


        return df


    @staticmethod
    def coerce_string_fields_from_model(df: pd.DataFrame, model_cls: Type) -> pd.DataFrame:
        """
        Coerce every column annotated as `str` (or `Optional[str]`) in the Pydantic `model_cls`
        to native Python strings. Ensures any <NA> values are converted to `None`.

        Produces concise debug reports via `logging.debug2` and `logging.debug3`.
        """
        logging.debug3("🔤 Coercing string fields from model...")
        df = df.copy()
        coerced_columns: dict[str, int] = {}
        nullified_columns: dict[str, int] = {}
        def _is_string(typ):
            origin = get_origin(typ)
            args   = get_args(typ)
            return typ is str or (origin in (Optional, Union) and str in args)

        for field_name, field in model_cls.model_fields.items():
            if field_name not in df.columns:
                logging.warning(f"❌ Column '{field_name}' not found in DataFrame, skipping string coercion.")
                continue

            if not _is_string(field.annotation):
                continue

            original = df[field_name]
            coerced  = original.apply(lambda x: str(x) if pd.notnull(x) else None)


            changed_mask = (original != coerced) & ~(original.isna() & pd.Series(coerced).isna())
            coerced_columns[field_name] = int(changed_mask.sum())

            if changed_mask.any():
                examples = original[changed_mask].dropna().astype(str).unique()[:3]
                logging.debug1(f"🔤 Column '{field_name}': coerced {changed_mask.sum()} "
                            f"values to `str`. Examples: {list(examples)}")
            else:
                logging.debug1(f"✅ Column '{field_name}': no string coercion needed")

            # Make sure all missing values are None
            na_before = pd.Series(coerced).isna().sum()
            coerced   = pd.Series(coerced).where(pd.notnull(coerced), None)
            na_after  = sum(val is None for val in coerced)

            total_coerced = sum(coerced_columns.values())
            total_nullified = sum(nullified_columns.values())

            if na_after:
                nullified_columns[field_name] = na_after
                logging.debug1(f"⚠️ Column '{field_name}': {na_after} <NA> replaced with None")

            df[field_name] = coerced

        logging.debug3(f"\n📋 String Coercion Complete: {total_coerced} coerced, {total_nullified} nullified")
        if coerced_columns:
            summary = ', '.join(f"{k}={v}" for k, v in coerced_columns.items())
            logging.debug2(f"🔤 String Coercion: {summary}")
        if nullified_columns:
            summary = ', '.join(f"{k}={v}" for k, v in nullified_columns.items())
            logging.debug2(f"⚠️ String Nulls: {summary}")


        return df





    @staticmethod
    def drop_incomplete_rows_from_model(
        df: pd.DataFrame,
        model_cls: Type[BaseModel],
    ) -> pd.DataFrame:
        """
        Remove every row that is *incomplete* with respect to the non-optional
        fields of `model_cls`.

        Parameters
        ----------
        df : pandas.DataFrame
            The dataframe to clean.
        model_cls : Type[pydantic.BaseModel]
            The Pydantic model that defines which columns are required.
        logger : logging.Logger | None, default None
            Logger used for the printed summary.  If *None* the root logger is
            used.

        Returns
        -------
        pandas.DataFrame
            A copy of *df* with incomplete rows removed.
        """

        # ------------------------------------------------------------------ helpers
        def _is_optional(ann) -> bool:
            """Return True if *ann* is typing.Optional[...] / Union[..., None]."""
            origin = get_origin(ann)
            if origin is Union:
                return type(None) in get_args(ann)
            return False

        # ---------------------------------------------------------------- identify required cols
        logging.debug3("🔍 Identifying required columns from model...")
        required_cols: list[str] = []
        for name, field in model_cls.model_fields.items():           # pydantic v2
            if _is_optional(field.annotation):
                continue
            # use alias if the dataframe uses it
            col_name = field.alias if field.alias else name
            required_cols.append(col_name)

        # Columns that are required but missing entirely in the frame
        missing_columns = [col for col in required_cols if col not in df.columns]
        if missing_columns:
            logging.warning(
                f"⚠️ Required column(s) {missing_columns} not present in DataFrame "
                f"— they will be ignored for completeness-check."
            )
            required_cols = [c for c in required_cols if c in df.columns]

        if not required_cols:
            logging.info("✅ No required columns found → no rows dropped.")
            return df.copy()

        # -------------------------------------------------------------- drop rows
        mask_incomplete = df[required_cols].isna().any(axis=1)
        dropped_count   = int(mask_incomplete.sum())
        kept_count      = len(df) - dropped_count

        cleaned_df = df.loc[~mask_incomplete].copy()

        # ------------------------------------------------------------------ summary
        logging.debug3(
            "\n📋 Completeness-check summary\n"
            f"   • Required columns : {required_cols}\n"
            f"   • Total rows       : {len(df)}\n"
            f"   • Dropped rows     : {dropped_count}\n"
            f"   • Remaining rows   : {kept_count}"
        )

        if dropped_count:
            # give a quick per-column breakdown (top 5)
            counts = (
                df.loc[mask_incomplete, required_cols]
                .isna()
                .sum()
                .sort_values(ascending=False)
                .head(5)
            )
            lines = "\n".join(f"     - {col}: {cnt} missing" for col, cnt in counts.items())
            logging.debug2("   • Top columns causing drops:\n" + lines)

        return cleaned_df

    @staticmethod
    def coerce_datetime_fields_from_model(
        df: pd.DataFrame,
        model_cls: Type[BaseModel],
    ) -> pd.DataFrame:
        """
        Coerces datetime fields in the DataFrame according to the model definition.
        Uses pandas.to_datetime with errors='coerce'.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame.
        model_cls : Type[pydantic.BaseModel]
            The model class defining expected datetime fields.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with datetime fields coerced.
        """

        from datetime import datetime
        df = df.copy()
        coerced_cols: dict[str, int] = {}
        failed_cols: dict[str, str] = {}
        missing_cols: list[str] = []

        def _is_datetime(ann) -> bool:
            origin = get_origin(ann)
            args = get_args(ann)
            return ann is datetime or (origin in (Union, Optional) and datetime in args)

        logging.debug3("📅 Coercing datetime fields from model...")

        for field_name, field in model_cls.model_fields.items():
            if not _is_datetime(field.annotation):
                continue

            col_name = field.alias if field.alias else field_name

            if col_name not in df.columns:
                missing_cols.append(col_name)
                logging.warning(f"❌ Column '{col_name}' not found in DataFrame — skipping datetime coercion.")
                continue

            try:
                original = df[col_name]
                coerced = pd.to_datetime(original, errors="coerce")
                invalid_count = (original.notna() & coerced.isna()).sum()

                df[col_name] = coerced
                coerced_cols[col_name] = int(invalid_count)

                if invalid_count > 0:
                    logging.debug1(f"⚠️ Column '{col_name}': {invalid_count} invalid datetime values coerced to NaT")
                else:
                    logging.debug1(f"✅ Column '{col_name}': datetime coercion successful")

            except Exception as e:
                failed_cols[col_name] = str(e)
                logging.error(f"❌ Failed to coerce column '{col_name}' to datetime: {e}")

        # -------------------------------------- Summary
        if coerced_cols:
            summary = ', '.join(f"{col}={cnt}" for col, cnt in coerced_cols.items())
            logging.debug2(f"\n📅 Datetime Coercion Summary: {summary}")

        if missing_cols:
            logging.debug2(
                f"\n⚠️ Missing datetime column(s) from model not found in DataFrame: {missing_cols}"
            )

        if failed_cols:
            summary = '\n'.join(f"   - {col}: {msg}" for col, msg in failed_cols.items())
            logging.debug2(f"\n⛔ Datetime Coercion Failures:\n{summary}")

        return df




    @staticmethod
    def drop_invalid_enum_rows_from_model(
        df: pd.DataFrame,
        model_cls: Type[BaseModel]
    ) -> pd.DataFrame:
        """
        Drops rows where Enum-annotated fields contain values not valid in the Enum.
        Logs per-column and total invalid value summaries.
        """
        df = df.copy()
        dropped_total = 0
        total_distinct_invalids = set()
        logging.debug3("🧹 Validating Enum fields from model...")

        for field_name, field in model_cls.model_fields.items():
            typ = field.annotation
            origin = get_origin(typ)
            args = get_args(typ)

            # Handle Optional[Enum] or Union[Enum, None]
            enum_cls = None
            if isinstance(typ, type) and issubclass(typ, Enum):
                enum_cls = typ
            elif origin in (Union, Optional):
                enum_args = [t for t in args if isinstance(t, type) and issubclass(t, Enum)]
                if enum_args:
                    enum_cls = enum_args[0]

            if enum_cls and field_name in df.columns:
                valid_values = set(e.value for e in enum_cls)
                col_series = df[field_name]
                invalid_mask = ~col_series.isin(valid_values) & col_series.notna()
                num_invalid = int(invalid_mask.sum())

                if num_invalid > 0:
                    dropped_total += num_invalid
                    invalid_vals = col_series[invalid_mask].dropna().unique()
                    distinct_count = len(invalid_vals)
                    total_distinct_invalids.update(invalid_vals)
                    examples = list(invalid_vals[:3])

                    logging.debug2(
                        f"  - Column '{field_name}': {num_invalid} invalid values, "
                        f"{distinct_count} distinct. Examples: {examples}"
                    )

                    df = df[~invalid_mask]
                else:
                    logging.debug3(f"  - Column '{field_name}': ✅ no invalid values")

        logging.debug3("\n🧹 Enum Validation Summary:")
        logging.debug2(
            f"\n🔎 Total distinct invalid enum values across all columns: "
            f"{len(total_distinct_invalids)}\n"
        )

        if dropped_total == 0:
            logging.debug2("✅ No rows dropped due to enum mismatches.")
        else:
            logging.debug2(f"✅ Dropped {dropped_total} rows with invalid Enum values.")

        return df


    @staticmethod
    def sanitize_columns_from_model(df: pd.DataFrame, model_cls: Type[BaseModel]) -> pd.DataFrame:
        """
        Drops all columns not defined in the model, and warns about missing ones.
        Returns a cleaned DataFrame and logs a summary.
        """
        logging.debug3("🧽 Sanitizing columns based on model...")
        df = df.copy()

        # Extract expected columns (with alias fallback)
        expected_cols = set(
            field.alias if field.alias else name
            for name, field in model_cls.model_fields.items()
        )
        df_cols = set(df.columns)

        # Identify columns to drop and columns that are missing
        extra_cols = df_cols - expected_cols
        missing_cols = expected_cols - df_cols

        # Drop extras
        df.drop(columns=list(extra_cols), inplace=True)

        # Logging summary
        logging.debug3(
            f"\n🧽 Column Sanitization Summary:\n"
            f"   • Dropped total of {len(extra_cols)} column(s)\n"
            f"     → {sorted(extra_cols) if extra_cols else 'None'}\n"
            f"   • Missing total of {len(missing_cols)} expected column(s)\n"
            f"     → {sorted(missing_cols) if missing_cols else 'None'}"
        )

        return df





    def sanitize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.to_object(df)
        df = self.to_python_scalars(df)
        df = self.replace_fakes(df)
        return df
