import re
import typing as t

from pydantic import validator

from pipeline.v3.schemas import BaseModel

POINTER_REGEX = re.compile(
    r"^[a-z0-9][a-z0-9-._/]*[a-z0-9]:[0-9A-Za-z_][0-9A-Za-z-_.]{0,127}$"
)


class PointerGet(BaseModel):
    id: str
    pointer: str
    pipeline_id: str
    locked: bool


class PointerCreate(BaseModel):
    pointer_or_pipeline_id: str
    pointer: str
    locked: t.Optional[bool]

    @validator("pointer")
    def validate_pointer(cls, v):
        if POINTER_REGEX.match(v) is None:
            raise ValueError("Invalid pointer name")
        return v


class PointerPatch(BaseModel):
    pointer_or_pipeline_id: t.Optional[str]
    locked: t.Optional[bool]
