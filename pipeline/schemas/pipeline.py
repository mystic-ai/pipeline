from datetime import datetime
from typing import Dict, List, Optional, Set

from pydantic import Field, root_validator, validator

from pipeline.schemas.base import BaseModel
from pipeline.schemas.compute_requirements import ComputeRequirements, ComputeType
from pipeline.schemas.environment import EnvironmentBrief
from pipeline.schemas.file import FileGet
from pipeline.schemas.function import FunctionGet
from pipeline.schemas.model import ModelGet
from pipeline.schemas.runnable import RunnableGet, RunnableType

from .validators import valid_pipeline_name, valid_pipeline_tag_name


class PipelineGraphNode(BaseModel):
    local_id: str
    function: str
    inputs: List[str]
    outputs: List[str]


class PipelineFileVariableGet(BaseModel):
    path: str
    hash: str
    file: FileGet


class PipelineVariableCreate(BaseModel):
    local_id: str
    name: Optional[str]

    type_file: Optional[FileGet] = Field(
        default=None,
        deprecated=True,
        description="Use multipart Pipeline creation instead.",
    )
    type_file_id: Optional[str] = Field(
        default=None,
        deprecated=True,
        description="Use multipart Pipeline creation instead.",
    )

    is_input: bool
    is_output: bool

    pipeline_file_variable: Optional[PipelineFileVariableGet]

    @root_validator
    def file_or_id_validation(cls, values):
        # If either deprecated field is set, verify that only one of them is set.
        type_file_defined = values.get("type_file") is not None
        type_file_id_defined = values.get("type_file_id") is not None
        deprecated_fields = type_file_defined or type_file_id_defined
        if deprecated_fields and type_file_defined == type_file_id_defined:
            raise ValueError(
                "Inline file must be set using `type_file` OR `type_file_id`."
            )
        return values


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
    environment: EnvironmentBrief


class PipelineCreate(BaseModel):
    name: str
    variables: List[PipelineVariableCreate]
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
    # Execution environment, defines Python dependencies
    environment_id: Optional[str]

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

    @validator("name")
    def validate_name(cls, value):
        if not valid_pipeline_name(value):
            raise ValueError(
                (
                    "May contain lowercase letters, digits and separators."
                    "Separators are periods, underscores, dashes and forward slashes."
                    "Can not start or end with a separator."
                )
            )
        return value


class PipelineTagCreate(BaseModel):
    # The full name of the tag, e.g. `my-pipeline:latest`.
    name: str
    # The pipeline ID this tag should point to.
    pipeline_id: str
    # The project ID is inferred from the project ID of the the pipeline.
    # project_id: str

    @validator("name")
    def validate_name(cls, value):
        if not valid_pipeline_tag_name(value):
            raise ValueError(
                (
                    "Must take the form: `name:tag`."
                    "Name must match the pipeline name."
                    "Tag may contain letters, digits, underscores, periods and dashes."
                    "Tag may contain a maximum of 128 characters."
                )
            )
        return value


class PipelineTagGet(BaseModel):
    id: str
    name: str
    project_id: str
    pipeline_id: str


class PipelineTagPatch(BaseModel):
    # The new pipeline ID this tag should point to.
    pipeline_id: str
