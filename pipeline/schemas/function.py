from typing import List, Optional

from pydantic import root_validator


from pipeline.schemas.base import BaseModel
from pipeline.schemas.file import FileGet, FileCreate


class FunctionBase(BaseModel):
    id: Optional[str]
    name: str


class FunctionGet(FunctionBase):
    id: str
    hex_file: FileGet
    # source_file_id: FileGet
    source_sample: str


class FunctionIO(BaseModel):
    """Descriptive schema of a function's input/output.

    Not intended to be sufficient to create an input/output type object.
    """

    # If present, the name of the argument
    name: Optional[str]
    # The name of the type, if available (some types cannot have their names
    # extracted)
    type_name: Optional[str]


class FunctionIOCreate(BaseModel):
    name: str
    file_id: Optional[str]
    file: Optional[FileCreate]

    @root_validator
    def file_or_id_validation(cls, values):
        file, file_id = values.get("file"), values.get("file_id")

        file_defined = file != None
        file_id_defined = file_id != None

        if file_defined == file_id_defined:
            raise ValueError(
                "You must define either the file OR file_id of a function."
            )

        return values


class FunctionCreate(BaseModel):
    local_id: str

    # function_hex: str
    function_source: str

    inputs: List[FunctionIO]
    output: List[FunctionIO]

    name: str
    hash: str

    file_id: Optional[str]
    file: Optional[FileCreate]

    @root_validator
    def file_or_id_validation(cls, values):
        file, file_id = values.get("file"), values.get("file_id")

        file_defined = file != None
        file_id_defined = file_id != None

        if file_defined == file_id_defined:
            raise ValueError(
                "You must define either the file OR file_id of a function."
            )

        return values
