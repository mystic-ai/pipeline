from pydantic import root_validator

from typing import List, Optional, Any, Union

from pipeline.schemas.base import BaseModel

from pipeline.schemas.file import FileCreate, FileGet
from pipeline.schemas.function import FunctionGet, FunctionCreate


class PipelineGraphNode(BaseModel):
    local_id: str
    function: str
    inputs: List[str]
    outputs: List[str]


class PipelineVariableGet(BaseModel):
    local_id: str
    name: Optional[str]

    type_file: Optional[FileGet]
    type_file_id: Optional[str]

    is_input: bool
    is_output: bool

    @root_validator
    def file_or_id_validation(cls, values):
        file, file_id = values.get("type_file"), values.get("type_file_id")

        file_defined = file != None
        file_id_defined = file_id != None

        if file_defined == file_id_defined:
            raise ValueError(
                "You must define either the type_file OR type_file_id of a variable."
            )

        return values


class PipelineGet(BaseModel):
    id: str
    name: str
    variables: List[PipelineVariableGet]
    functions: List[FunctionGet]
    graph_nodes: List[PipelineGraphNode]
    outputs: List[str]

    class Config:
        orm_mode = True


class PipelineCreate(BaseModel):
    name: str
    variables: List[PipelineVariableGet]
    functions: List[FunctionGet]
    graph_nodes: List[PipelineGraphNode]
    outputs: List[str]
    # models: Optional[dict]
