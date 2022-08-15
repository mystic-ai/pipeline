from datetime import datetime
from typing import Dict, List, Optional, Set

from pydantic import Field, root_validator, validator

from pipeline.schemas.base import BaseModel
from pipeline.schemas.compute_requirements import ComputeRequirements, ComputeType
from pipeline.schemas.file import FileGet
from pipeline.schemas.function import FunctionGet
from pipeline.schemas.model import ModelGet
from pipeline.schemas.runnable import RunnableGet, RunnableType


class PipelineGraphNode(BaseModel):
    local_id: str
    function: str
    inputs: List[str]
    outputs: List[str]


class PipelineFileVariableGet(BaseModel):
    path: str
    hash: str
    file: FileGet


class PipelineVariableGet(BaseModel):
    local_id: str
    name: Optional[str]

    type_file: Optional[FileGet]
    type_file_id: Optional[str]

    is_input: bool
    is_output: bool

    pipeline_file_variable: Optional[PipelineFileVariableGet]

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


class PipelineGetBrief(BaseModel):
    id: str
    name: str
    deployed: bool = False
    tags: List[str] = []
    description: str = ""


class PipelineGet(PipelineGetBrief, RunnableGet):
    type: RunnableType = Field(RunnableType.pipeline, const=True)
    variables: List[PipelineVariableGet]
    functions: List[FunctionGet]
    models: List[ModelGet]
    graph_nodes: List[PipelineGraphNode]
    outputs: List[str]

    class Config:
        orm_mode = True


class PipelineGetDetailed(PipelineGet):
    version: str = "1"
    dependencies: List[str] = ["torch", "transformers"]
    created_at: datetime
    updated_at: datetime
    last_run: Optional[datetime]
    public: bool
    # Maps language, e.g. `curl` or `python`, to an example Run creation code snippet
    run_examples: Dict[str, str] = {}


class PipelineCreate(BaseModel):
    name: str
    variables: List[PipelineVariableGet]
    functions: List[FunctionGet]
    models: List[ModelGet]
    graph_nodes: List[PipelineGraphNode]
    outputs: List[str]
    project_id: Optional[str]
    public: bool = False
    description: str = ""
    tags: Set[str] = set()
    # By default a Pipeline will require GPU resources
    compute_type: ComputeType = ComputeType.gpu
    compute_requirements: Optional[ComputeRequirements]

    @validator("compute_requirements")
    def compute_type_is_gpu(cls, v, values):
        """If compute_type is not 'gpu' we don't expect any additional compute
        requirements to be specified.
        """
        if values["compute_type"] != ComputeType.gpu:
            if v and v.min_gpu_vram_mb:
                raise ValueError(
                    "min_gpu_vram_mb should only be specified for gpu workloads"
                )
        return v
