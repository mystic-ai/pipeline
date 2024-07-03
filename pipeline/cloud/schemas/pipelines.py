import typing as t
from datetime import datetime
from enum import Enum

from pipeline.cloud.compute_requirements import Accelerator
from pipeline.cloud.schemas import BaseModel, pagination
from pipeline.cloud.schemas.cluster import PipelineClusterConfig, PipelineClusterGetLean
from pipeline.cloud.schemas.runs import RunIOType


class IOVariable(BaseModel):
    run_io_type: RunIOType

    title: str | None
    description: str | None
    examples: list[t.Any] | None
    gt: int | None
    ge: int | None
    lt: int | None
    le: int | None
    multiple_of: int | None
    allow_inf_nan: bool | None
    max_digits: int | None
    decimal_places: int | None
    min_length: int | None
    max_length: int | None
    choices: list[t.Any] | None
    dict_schema: t.List[dict] | None
    default: t.Any | None
    optional: bool | None


class PipelineStartUpload(BaseModel):
    pipeline_name: str
    pipeline_tag: t.Optional[str]
    cluster: PipelineClusterConfig | None = None


class PipelineStartUploadResponse(BaseModel):
    bearer: str
    upload_registry: t.Optional[str]
    #: Full username/pipeline_name. Used for naming docker image
    pipeline_name: str


class PipelineCreate(BaseModel):
    name: str
    image: str

    input_variables: t.List[IOVariable]
    output_variables: t.List[IOVariable]

    accelerators: t.Optional[t.List[Accelerator]]

    cluster: PipelineClusterConfig | None = None

    scaling_config: str | None = None

    # Additional meta data
    description: t.Optional[str]
    readme: t.Optional[str]
    extras: t.Optional[dict]


class Pipeline(BaseModel):
    name: str
    image: str

    input_variables: t.List[IOVariable]
    output_variables: t.List[IOVariable]
    extras: t.Optional[dict]


class PipelinePatch(BaseModel):
    input_variables: t.Optional[t.List[IOVariable]]
    output_variables: t.Optional[t.List[IOVariable]]

    accelerators: t.Optional[t.List[Accelerator]]

    extras: t.Optional[dict]
    #: The name of the scaling configuration
    scaling_config: str | None = None


class PipelineListPagination(pagination.Pagination):
    class OrderBy(str, Enum):
        created_at = "created_at"
        updated_at = "updated_at"

        pipeline_name = "pipeline_name"
        image = "image"

    order_by: OrderBy
    order: pagination.Order


class PipelineDeploymentStatus(str, Enum):
    not_deployed = "not_deployed"
    deploying = "deploying"
    deployed = "deployed"
    failed = "failed"
    deleting = "deleting"
    deleted = "deleted"


class PipelineState(str, Enum):
    not_loaded = "not_loaded"
    loading = "loading"
    loaded = "loaded"
    load_failed = "load_failed"
    startup_failed = "startup_failed"
    container_failed = "container_failed"

    # backwards compatability
    failed = "failed"


class PipelineContainerState(BaseModel):
    state: PipelineState
    message: t.Optional[str]


class PipelineScalingInfo(BaseModel):
    current_replicas: int
    desired_replicas: int
    current_pipeline_states: dict[PipelineState, int]


class PipelineGet(Pipeline):
    id: str

    created_at: datetime
    updated_at: datetime

    accelerators: t.Optional[t.List[Accelerator]]

    cluster: PipelineClusterGetLean | None = None

    extras: t.Optional[dict]
    #: The name of the scaling configuration
    scaling_config: str | None = None
    #: Additional info attached when pipeline fails to load or startup
    failed_state_info: t.Optional[PipelineContainerState]
