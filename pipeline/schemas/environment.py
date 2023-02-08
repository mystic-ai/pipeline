from datetime import datetime
from typing import List, Optional

from pipeline.schemas.base import BaseModel


class EnvironmentCreate(BaseModel):
    name: str
    python_requirements: List[str]


class EnvironmentBrief(BaseModel):
    id: str
    name: str
    deleted_at: Optional[datetime]


class EnvironmentGet(BaseModel):
    id: str
    name: str
    python_requirements: List[str]
    locked: bool


class EnvironmentPatch(BaseModel):
    python_requirements: Optional[List[str]]
    locked: Optional[bool]
