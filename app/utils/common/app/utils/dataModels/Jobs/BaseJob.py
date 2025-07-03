from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Union, Dict, ClassVar
from uuid import UUID, uuid4
from datetime import datetime, timezone
from enum import Enum
import json
import math

from app.utils.common.app.utils.dataModels.Jobs.util.RetryInfo import RetryInfo

class BaseJob(BaseModel):
    model_config = ConfigDict(extra="forbid")
    orm_model: ClassVar[Optional[type]] = None  # REpresents the underlying orm_api this gets set in subclasses


    job_uuid : str = Field(default_factory=uuid4)
    job_type : str


    status  : str
    attempts: int = 0
    next_retry: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    created  : datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated  : datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    parent_job_uuids: List[str] = Field(default_factory=list)

    def register_failure(self, error: str, penalty_step: float = 1.0, backoff: float = 1.0):
        self.attempts += 1
        self.next_retry = RetryInfo.compute_next_retry(
            attempts=self.attempts,
            baseline=self.next_retry,
            penalty_step=penalty_step,
            backoff=backoff,
        )
        self.updated = datetime.now(timezone.utc)

    def update_db(self, fields_to_update: Optional[List[str]] = None):
        if self.orm_model is None:
            raise ValueError(f"{self.__class__.__name__} must define `orm_model` to support update_db()")

        self.updated = datetime.now(timezone.utc)

        if fields_to_update is not None and "payload" not in fields_to_update:
            # Fast path: minimal row update
            row = {
                "job_uuid": self.job_uuid,
                "updated": self.updated,
            }
            for f in fields_to_update:
                if f not in {"job_uuid", "updated"}:
                    row[f] = getattr(self, f, None)
        else:
            # Full row update
            row = self.to_sql_row()
            row["updated"] = self.updated
            if fields_to_update is not None:
                row = {k: v for k, v in row.items() if k in fields_to_update or k in {"job_uuid", "updated"}}

        self.orm_model.update_row(row)

    def update_timestamp(self):
        self.updated = datetime.now(timezone.utc)

        
    def is_ready(self, task: str) -> bool:
        task_field = f"{task}_status"
        if not hasattr(self, task_field):
            raise ValueError(f"Unknown task: {task}")

    def to_orm(self, orm_cls):
        return orm_cls(**self.to_sql_row())

    def to_sql_row(self) -> dict:
        def _clean_nans(obj):
            """
            Recursively replace all float('nan'), inf, -inf with None in a nested structure.
            This ensures the object can be safely serialized to valid JSON.
            """
            if isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return None
                return obj
            elif isinstance(obj, dict):
                return {k: _clean_nans(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_clean_nans(item) for item in obj]
            return obj


        base_fields = {}
        raw_dict = self.model_dump(mode="json")     # Pydantic v2
        clean_dict = _clean_nans(raw_dict)
        for field_name, value in clean_dict.items():
            if isinstance(value, Enum):
                base_fields[field_name] = value.value
            elif isinstance(value, (UUID, datetime)):
                base_fields[field_name] = str(value)
            else:
                base_fields[field_name] = value

        payload = {
            "input": clean_dict.get("input", {}),
            "attrs": clean_dict.get("attrs", {})
        }

        return {
            "job_uuid": self.job_uuid,
            "job_type": self.job_type,
            "status": self.status,
            "attempts": self.attempts,
            "next_retry": self.next_retry,
            "created": self.created,
            "updated": self.updated,
            "parent_job_uuids": self.parent_job_uuids,
            "payload": payload
        }



    @classmethod
    def from_sql_row(cls, row: dict) -> "BaseJob":
        payload = row.get("payload", {}) or {}

        merged = {
            "job_uuid": row.get("job_uuid"),     # explicit
            **payload,
            "status": row.get("status"),
            "created": row.get("created"),
            "updated": row.get("updated"),
            "attempts": row.get("attempts", 0),
            "next_retry": row.get("next_retry", datetime.now(timezone.utc)),
            "job_type": row.get("job_type"),
            "parent_job_uuids": row.get("parent_job_uuids", []),
        }
        return cls.model_validate(merged)



