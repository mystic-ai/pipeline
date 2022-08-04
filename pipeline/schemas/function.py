from typing import List, Optional

from pydantic import Field, root_validator

from pipeline.schemas.base import BaseModel
from pipeline.schemas.compute_requirements import ComputeRequirements, ComputeType
from pipeline.schemas.file import FileCreate, FileGet
from pipeline.schemas.runnable import RunnableGet, RunnableType


class FunctionBase(BaseModel):
    id: Optional[str]
    name: str


class FunctionGet(RunnableGet):
    id: str
    hex_file: FileGet

    source_sample: str
    type: RunnableType = Field(RunnableType.function, const=True)

    class Config:
        orm_mode = True


class FunctionIO(BaseModel):
    """Descriptive schema of a function's input/output.

    Not intended to be sufficient to create an input/output type object.
    """

    # If present, the name of the argument
    name: str
    # The name of the type, if available (some types cannot have their names
    # extracted)
    type_name: str


class FunctionGetDetailed(FunctionGet):
    inputs: List[FunctionIO]
    output: List[FunctionIO]


class FunctionIOCreate(BaseModel):
    name: str
    file_id: Optional[str]
    file: Optional[FileCreate]

    @root_validator
    def file_or_id_validation(cls, values):
        file, file_id = values.get("file"), values.get("file_id")

        file_defined = file is not None
        file_id_defined = file_id is not None

        if file_defined == file_id_defined:
            raise ValueError(
                "You must define either the file OR file_id of a function."
            )

        return values


class FunctionCreate(BaseModel):
    # The local ID is assigned when a new function is used as part of a new
    # pipeline; the server uses the local ID to associated a function to a
    # graph node before replacing the local ID with the server-generated one
    local_id: Optional[str]

    # function_hex: str
    function_source: str

    inputs: List[FunctionIO]
    output: List[FunctionIO]

    name: str
    hash: str

    file_id: Optional[str]
    file: Optional[FileCreate]

    # By default a Function will require GPU resources
    compute_type: ComputeType = ComputeType.gpu
    compute_requirements: Optional[ComputeRequirements]

    @root_validator
    def file_or_id_validation(cls, values):
        file, file_id = values.get("file"), values.get("file_id")

        file_defined = file is not None
        file_id_defined = file_id is not None

        if file_defined == file_id_defined:
            raise ValueError(
                "You must define either the file OR file_id of a function."
            )

        return values
