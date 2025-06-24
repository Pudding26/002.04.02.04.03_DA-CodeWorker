from pydantic import BaseModel
from typing import Optional, Dict

class TaskRequestBase(BaseModel):
    task_name: str
    custom_task_name: Optional[str] = None
    params: Optional[Dict] = None

class TaskResponseBase(BaseModel):
    status: str
    task: str
