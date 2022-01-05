import datetime
from enum import Enum
from typing import List, Optional, Union

from pydantic import root_validator

from pipeline.schemas.file import FileGet
from pipeline.schemas.function import FunctionGet, FunctionGetDetailed
from pipeline.schemas.pipeline import PipelineGet, PipelineGetDetailed

from .base import BaseModel
from .data import DataGet
from .runnable import RunnableIOGet
from .tag import TagGet
from .token import TokenGet


class RunState(Enum):
    RECEIVED = "received"
    ALLOCATING_CLUSTER = "allocating_cluster"
    AWAITING_RESOURCE_ALLOCATION = "awaiting_resource_allocation"
    ALLOCATING_RESOURCE = "allocating_resource"
    LOADING_DATA = "loading_data"
    LOADING_FUNCTION = "loading_function"
    LOADING_PIPELINE = "loading_pipeline"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class RunError(Enum):
    ...


class RunCreate(BaseModel):
    pipeline_id: Optional[str]
    function_id: Optional[str]
    data: Optional[str]
    data_id: Optional[str]
    blocking: Optional[bool] = False

    @root_validator
    def pipeline_data_val(cls, values):
        pipeline_id, function_id = values.get("pipeline_id"), values.get("function_id")

        pipeline_defined = pipeline_id is not None
        function_defined = function_id is not None

        if pipeline_defined == function_defined:
            raise ValueError("You must define either a pipeline_id OR function_id.")

        data_id, data = values.get("data_id"), values.get("data")

        data_defined = data is not None
        data_id_defined = data_id is not None

        if data_defined == data_id_defined:
            raise ValueError("You must define either a data_id OR data.")

        return values


class RunIOGet(RunnableIOGet):
    """Realised/given input/output data to a Runnable used in a Run."""

    value: str
    data_url: str


class RunGet(BaseModel):
    id: str
    created_at: datetime.datetime
    started_at: Optional[datetime.datetime]
    ended_at: Optional[datetime.datetime]
    run_state: RunState
    compute_time_ms: Optional[int]
    runnable: Union[FunctionGet, PipelineGet]
    data: DataGet
    blocking: Optional[bool] = False
    result: Optional[FileGet]
    error: Optional[RunError]

    class Config:
        allow_population_by_field_name = True


class RunGetDetailed(RunGet):
    runnable: Union[FunctionGetDetailed, PipelineGetDetailed]
    n_resources: int
    resource_type: str
    region: str
    tags: List[TagGet] = []
    inputs: List[RunIOGet] = []
    outputs: List[RunIOGet] = []
    token: TokenGet


class RunUpdate(BaseModel):
    result_id: Optional[str]
    run_state: Optional[RunState]
    compute_cluster_id: Optional[str]
