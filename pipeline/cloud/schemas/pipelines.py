import typing as t
from datetime import datetime

from pipeline.cloud.compute_requirements import Accelerator
from pipeline.cloud.schemas import BaseModel
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


class PipelineGet(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime

    name: str
    path: str

    minimum_cache_number: t.Optional[int]
    gpu_memory_min: t.Optional[int]
    environment_id: str

    accelerators: t.Optional[t.List[Accelerator]]

    input_variables: t.List[IOVariable]
    output_variables: t.List[IOVariable]

    _metadata: t.Optional[dict]


class PipelinePatch(BaseModel):
    minimum_cache_number: t.Optional[int]
    gpu_memory_min: t.Optional[int]
    accelerators: t.Optional[t.List[Accelerator]]


########## v4 Schemas ##########


class PipelineStartUpload(BaseModel):
    pipeline_name: str
    pipeline_tag: t.Optional[str]


class PipelineStartUploadResponse(BaseModel):
    bearer: str
    upload_registry: t.Optional[str]


class PipelineCompleteUpload(BaseModel):
    name: str
    container_hash: str

    input_variables: t.List[IOVariable]
    output_variables: t.List[IOVariable]

    vram_mb: t.Optional[int]
    minimum_cache_number: t.Optional[int]
    accelerators: t.Optional[t.List[Accelerator]]

    _metadata: t.Optional[dict]


class PipelineContainerGet(BaseModel):
    id: str

    created_at: datetime
    updated_at: datetime

    name: str

    minimum_cache_number: t.Optional[int]
    gpu_memory_min: t.Optional[int]

    accelerators: t.Optional[t.List[Accelerator]]

    input_variables: t.List[IOVariable]
    output_variables: t.List[IOVariable]

    _metadata: t.Optional[dict]
