from typing import List, Optional

from pydantic import Field, root_validator

from pipeline.schemas.base import BaseModel
from pipeline.schemas.file import FileGet
from pipeline.schemas.function import FunctionGet
from pipeline.schemas.model import ModelGet
from pipeline.schemas.runnable import RunnableGet, RunnableType


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

        file_defined = file is not None
        file_id_defined = file_id is not None

        if file_defined == file_id_defined:
            raise ValueError(
                "You must define either the type_file OR type_file_id of a variable."
            )

        return values


class PipelineGet(RunnableGet):
    id: str
    name: str
    type: RunnableType = Field(RunnableType.pipeline, const=True)
    variables: List[PipelineVariableGet]
    functions: List[FunctionGet]
    models: List[ModelGet]
    graph_nodes: List[PipelineGraphNode]
    outputs: List[str]

    class Config:
        orm_mode = True


class PipelineGetDetailed(PipelineGet):
    ...


class PipelineCreate(BaseModel):
    name: str
    variables: List[PipelineVariableGet]
    functions: List[FunctionGet]
    models: List[ModelGet]
    graph_nodes: List[PipelineGraphNode]
    outputs: List[str]
