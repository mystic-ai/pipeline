from typing import Optional

from pydantic import Field, root_validator

from pipeline.schemas.base import BaseModel
from pipeline.schemas.file import FileCreate, FileGet


class ModelBase(BaseModel):
    id: str
    name: str


class ModelGet(ModelBase):
    hex_file: FileGet

    source_sample: str

    class Config:
        orm_mode = True


class ModelGetDetailed(ModelGet):
    ...


class ModelGetOverview(ModelBase):
    description: str
    pipeline_count: int


class ModelCreate(BaseModel):
    # The local ID is assigned when a new model is used as part of a new
    # pipeline; the server uses the local ID to associated a model to a
    # Pipeline before replacing the local ID with the server-generated one
    local_id: Optional[str]

    model_source: str
    hash: str
    name: str

    file_id: Optional[str] = Field(
        default=None,
        deprecated=True,
        description="Use multipart Model creation instead.",
    )
    file: Optional[FileCreate] = Field(
        default=None,
        deprecated=True,
        description="Use multipart Model creation instead.",
    )

    @root_validator
    def file_or_id_validation(cls, values):
        # If either deprecated field is set, verify that only one of them is set.
        file_defined = values.get("file") is not None
        file_id_defined = values.get("file_id") is not None
        deprecated_fields = file_defined or file_id_defined
        if deprecated_fields and file_defined == file_id_defined:
            raise ValueError("Inline file must be set using `file` OR `file_id`.")
        return values
