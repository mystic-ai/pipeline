import datetime
from enum import Enum
from typing import Any, List, Optional, Union

from pydantic import root_validator, validator

from pipeline.schemas.compute_requirements import ComputeRequirements, ComputeType
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


class RunErrorType(Enum):
    """The type of error that occurred (if any)"""

    MAX_RETRIES = "max_retries"
    PIPELINE_FAULT = "pipeline_fault"
    UNSATISFIABLE = "unsatisfiable"


class RunErrorInfo(BaseModel):
    """More info about the error if it was a pipeline_fault"""

    exception: str
    traceback: Optional[str]


class RunCreate(BaseModel):
    pipeline_id: Optional[str]
    function_id: Optional[str]
    data: Optional[Any]
    data_id: Optional[str]
    blocking: Optional[bool] = False
    compute_type: Optional[ComputeType]
    compute_requirements: Optional[ComputeRequirements]

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
    resource_type: Optional[str]
    compute_time_ms: Optional[int]
    runnable: Union[FunctionGet, PipelineGet]
    data: DataGet
    blocking: Optional[bool] = False
    result: Optional[FileGet]
    #: JSON-serialised runnable return value, if available
    result_preview: Optional[Union[list, dict]]
    error: Optional[RunErrorType]
    error_info: Optional[RunErrorInfo]
    compute_requirements: Optional[ComputeRequirements] = None

    class Config:
        allow_population_by_field_name = True

    @validator("compute_requirements", pre=True)
    def compute_requirements_default_if_empty(cls, v):
        """Return None if no compute_requirements rather than an empty
        ComputeRequirements object
        """
        if not v:
            return None
        return v


class RunGetDetailed(RunGet):
    runnable: Union[FunctionGetDetailed, PipelineGetDetailed]
    n_resources: int
    region: str
    tags: List[TagGet] = []
    inputs: List[RunIOGet] = []
    outputs: List[RunIOGet] = []
    #: The Token which was used to create this Run, if that Token has not been
    #: deleted in the meantime
    token: Optional[TokenGet]


class RunUpdate(BaseModel):
    result_id: Optional[str]
    run_state: Optional[RunState]
    compute_cluster_id: Optional[str]
