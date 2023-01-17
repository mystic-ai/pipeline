from typing import List

from pipeline.schemas.base import BaseModel


class EnvironmentCreate(BaseModel):
    name: str
    python_requirements: List[str]


class EnvironmentGet(BaseModel):
    id: str
    name: str
    python_requirements: List[str]
    is_locked: bool


class EnvironmentPythonRequirementsUpdate(BaseModel):
    python_requirements: List[str]
