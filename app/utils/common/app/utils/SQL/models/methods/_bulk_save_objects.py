"""bulk_save_objects.py – Error‑handling overhaul
------------------------------------------------
* Accepts an optional *logging* parameter so you can inject your own logging system.
* Sanitises DB‑API/SQLAlchemy exception strings to avoid huge SQL payloads.
* Never re‑raises the original spammy exception – instead raises a concise *BulkInsertError* summarising failures.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple, Type

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, class_mapper
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


__all__ = ["_bulk_save_objects", "BulkInsertError"]


class BulkInsertError(RuntimeError):
    """Raised when the abort_threshold is exceeded.

    Attributes
    ----------
    failed_rows : List[Tuple[int, str]]
        (row_index, reason) tuples for each row that could not be inserted.
    """

    def __init__(self, failed_rows: List[Tuple[int, str]]):
        self.failed_rows = failed_rows
        message = (
            f"Abort threshold reached – {len(failed_rows)} rows failed. "
            "Check `failed_rows` for details."
        )
        super().__init__(message)


# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------


def _sanitize_exc(exc: SQLAlchemyError, max_len: int = 200) -> str:  # noqa: D401
    """Return a single‑line, trimmed representation of *exc* (max *max_len* chars)."""

    # Prefer the underlying DB‑API error, it is usually shorter.
    raw = str(getattr(exc, "orig", exc))
    single_line = raw.replace("\n", " ").strip()
    return (single_line[: max_len - 3] + "…") if len(single_line) > max_len else single_line


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------


def _bulk_save_objects(
    orm_cls: Type,  # SQLAlchemy declarative model
    records: List[Dict[str, Any]],
    session: Session,
    *,
    batch_size: int = 5000,
    switch_threshold: int = 5,
    abort_threshold: int = 15,
) -> None:
    """Insert *records* with graceful fallbacks and concise diagnostics.

    Parameters
    ----------
    orm_cls : SQLAlchemy declarative model class.
    records : List of row dictionaries.
    session : Active ``Session`` instance.
    batch_size : Bulk chunk size.
    switch_threshold : Row‑failures in a chunk before cell probing starts.
    abort_threshold : Global failure ceiling before *BulkInsertError* is raised.
    logging : Optional custom logging; defaults to ``logging.getLogger(__name__)``.
    """


    # Cache required columns (non‑nullable **or** PK)
    required_cols = {
        col.name for col in class_mapper(orm_cls).columns if (not col.nullable) or col.primary_key
    }

    # Track failures to summarise later
    failed_rows: List[Tuple[int, str]] = []

    # --------------------------------------------------------------
    # Helper: column‑by‑column probe inside SAVEPOINT
    # --------------------------------------------------------------
    def _probe_row(row_index: int, row_data: Dict[str, Any]) -> None:
        for column, value in row_data.items():
            if column in required_cols:
                continue
            test_payload = {k: (v if (k in required_cols or k == column) else None)
                            for k, v in row_data.items()}

            try:
                with session.begin_nested():
                    session.add(orm_cls(**test_payload))
                    session.flush()
                session.rollback()
                logging.debug2("✅ CELL OK  @ row %d → column '%s'", row_index, column)
            except SQLAlchemyError:
                session.rollback()
                logging.warning("🔬 CELL FAIL @ row %d → column '%s' → value: %s",
                                row_index, column, repr(value))
                break


    # --------------------------------------------------------------
    # Phase 1 – Bulk insert loop
    # --------------------------------------------------------------

    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        objects = [orm_cls(**row) for row in chunk]

        try:
            session.bulk_save_objects(objects)
            session.commit()
            logging.debug2("✅ BULK OK – rows %d … %d", start, start + len(objects) - 1)
            continue
        except SQLAlchemyError as exc:
            session.rollback()
            logging.warning(
                "⚠️ Bulk insert failed (batch starts row %d) – %s → switching to row mode.",
                start,
                _sanitize_exc(exc),
            )


        # ----------------------------------------------------------
        # Phase 2 – Row‑by‑row fallback for this chunk
        # ----------------------------------------------------------
        
        # Dynamically recreate engine with echo=True
        #original_url = str(session.get_bind().url)
        #debug_engine = create_engine(original_url, echo=True)
        #
        #logging.basicConfig()
        #logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
        #
        #DebugSession = sessionmaker(bind=debug_engine)
        #session = DebugSession()
        
        row_failures_in_chunk = 0

        for offset, row in enumerate(chunk):
            idx = start + offset
            try:
                session.add(orm_cls(**row))
                session.flush()
                session.commit()
                logging.debug("✅ Row %d OK.", idx)
            except SQLAlchemyError as exc_row:
                session.rollback()
                reason = _sanitize_exc(exc_row)
                failed_rows.append((idx, reason))
                row_failures_in_chunk += 1
                logging.debug("⚠️ Row %d failed – %s", idx, reason)

                if row_failures_in_chunk >= switch_threshold:
                    _probe_row(idx, row)

                if len(failed_rows) >= abort_threshold:
                    logging.error(
                        "🚨 ABORT threshold hit (%d failures). Raising BulkInsertError.",
                        len(failed_rows),
                    )
                    raise BulkInsertError(failed_rows) from None

    # --------------------------------------------------------------
    # Completed without hitting abort_threshold
    # --------------------------------------------------------------
    if failed_rows:
        logging.warning("Completed with %d problem rows.", len(failed_rows))
    else:
        logging.info("🎉 All rows inserted successfully.")
