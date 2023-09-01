from datetime import datetime
from enum import Enum

from pipeline.cloud.schemas import BaseModel


class EnvironmentVerificationStatus(str, Enum):
    unverified = "unverified"
    verifying = "verifying"
    verified = "verified"
    failed = "failed"


class EnvironmentGet(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime

    name: str
    python_requirements: list[str]
    hash: str

    verification_status: EnvironmentVerificationStatus
    verification_exception: str | None
    verification_traceback: str | None


class EnvironmentCreate(BaseModel):
    name: str
    python_requirements: list[str]
