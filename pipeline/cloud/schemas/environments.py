from datetime import datetime

from pipeline.cloud.schemas import BaseModel


class EnvironmentGet(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime

    name: str
    python_requirements: list[str]
    hash: str


class EnvironmentCreate(BaseModel):
    name: str
    python_requirements: list[str]
