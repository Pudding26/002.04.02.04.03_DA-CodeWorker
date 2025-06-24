import os
import logging
from time import time
from typing import Type, List
from collections import Counter, defaultdict

import pandas as pd
from pydantic import BaseModel, ValidationError
from pydantic import TypeAdapter


def _model_validate_dataframe(
    cls: Type[BaseModel],
    df: pd.DataFrame,
    batch_size: int = 5000,
    max_errors: int = 15
) -> List[dict]:
    """
    Validate a dataframe against a Pydantic model in batch or row-wise mode.

    Args:
        cls: The Pydantic model class to validate against.
        df: The input DataFrame.
        batch_size: Number of rows per validation batch.
        max_errors: Maximum number of errors before aborting early.

    Returns:
        A list of validated rows (as dicts) where `None` replaces missing values.
    """
    adapter = TypeAdapter(list[cls])
    validated_records = []

    error_counter = Counter()
    column_counter = Counter()
    column_error_map = defaultdict(lambda: Counter())

    DEBUG_PYDANTIC = os.getenv("DEBUG_PYDANTIC", True)

    rows_failed = 0
    rows_passed = 0
    start = time()
    no_batches = max(1, len(range(1, len(df), batch_size)))
    max_errors = max_errors * no_batches

    for i in range(0, len(df), batch_size):
        chunk = df.iloc[i:i + batch_size]

        if DEBUG_PYDANTIC in ("True", True):
            logging.debug2("🔍 PydanticValidator detected Debug Flag (row-wise mode)")
            for idx, row in chunk.iterrows():
                try:
                    validated = cls.model_validate(row.to_dict())
                    validated_records.append(validated)
                    rows_passed += 1
                except ValidationError as ve:
                    rows_failed += 1
                    _record_row_errors(
                        ve, row, idx, column_counter, column_error_map
                    )
                    logging.warning(f"⚠️ Validation error in row {idx}: {ve}")
                    if rows_failed >= max_errors:
                        logging.error(
                            f"❌ Stopping validation early: exceeded max error threshold ({max_errors})"
                        )
                        break
        else:
            try:
                validated = adapter.validate_python(chunk.to_dict(orient="records"))
                validated_records.extend(validated)
                rows_passed += len(validated)
            except ValidationError as ve:
                rows_failed += len(chunk)
                _record_batch_errors(ve, column_counter, column_error_map, error_counter)
                logging.warning(f"⚠️ Validation error in rows {i}–{i + batch_size}: {ve}")
                if rows_failed >= max_errors:
                    logging.error(
                        f"❌ Stopping validation early: exceeded max error threshold ({max_errors})"
                    )
                    break

    end = time()
    _log_validation_summary(
        start, end, rows_passed, rows_failed,
        column_counter, column_error_map, error_counter
    )

    # Return only clean dicts with nulls removed
    return [record.model_dump(exclude_none=True) for record in validated_records]


def _record_row_errors(
    ve: ValidationError,
    row: pd.Series,
    row_idx: int,
    column_counter: Counter,
    column_error_map: defaultdict
) -> None:
    """
    Extract and log validation error details from a single row.

    Adds error counts to column and type maps.
    """
    for error in ve.errors():
        loc = error["loc"]
        err_type = error["type"]
        col = loc[0] if isinstance(loc, tuple) else loc
        value = row.get(col, None)

        column_counter[col] += 1
        column_error_map[col][err_type] += 1

        # Save a few example values for context
        if "examples" not in column_error_map[col]:
            column_error_map[col]["examples"] = []
        if len(column_error_map[col]["examples"]) < 3:
            column_error_map[col]["examples"].append(value)


def _record_batch_errors(
    ve: ValidationError,
    column_counter: Counter,
    column_error_map: defaultdict,
    error_counter: Counter
) -> None:
    """
    Extract and log validation error details from a batch of rows.
    """
    for error in ve.errors():
        loc = error["loc"]
        err_type = error["type"]
        col = loc[0] if isinstance(loc, tuple) else loc

        column_counter[col] += 1
        column_error_map[col][err_type] += 1
        error_counter[err_type] += 1



def _log_validation_summary(
    start: float,
    end: float,
    rows_passed: int,
    rows_failed: int,
    column_counter: Counter,
    column_error_map: defaultdict,
    error_counter: Counter
) -> None:
    """
    Emit detailed debug2-level summary of validation results.
    """
    logging.debug2("📋 Validation summary:")
    logging.debug2(f"   ⏱️ Time taken: {end - start:.2f} sec")
    logging.debug2(f"   ✅ Rows passed: {rows_passed}")
    logging.debug2(f"   ❌ Rows failed: {rows_failed} (out of {rows_passed + rows_failed})")

    if rows_failed == 0:
        return

    logging.debug2(f"   🔁 Columns with issues: {dict(column_counter.most_common(5))}")
    logging.debug2(f"   🔍 Top error types: {dict(error_counter.most_common(5))}")

    sorted_columns = sorted(
        column_error_map.items(),
        key=lambda x: sum(v for k, v in x[1].items() if k != "examples"),
        reverse=True
    )

    for col, errors in sorted_columns:
        total_issues = sum(v for k, v in errors.items() if k != "examples")
        logging.debug2(f"   🧩 Column '{col}' had {total_issues} validation issue(s):")
        for err_type, count in errors.items():
            if err_type == "examples":
                examples = ', '.join(str(x) for x in count)
                logging.debug2(f"      🧪 Example values: {examples}")
            else:
                logging.debug2(f"      ↳ {err_type}: {count}×")
