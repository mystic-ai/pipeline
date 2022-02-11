import enum
from datetime import datetime, timedelta
from typing import List, Optional

from pydantic import Field

from .base import BaseModel
from .project import ProjectGet


class RunnableType(enum.Enum):
    function = "function"
    pipeline = "pipeline"


class RunnableIOGet(BaseModel):
    """Unrealised/expected input/output data to a Runnable."""

    name: str
    type: str


class RunnableGet(BaseModel):
    id: str
    type: str
    name: Optional[str]
    project: ProjectGet


class RunnableGetDetailed(RunnableGet):
    expected_inputs: List[RunnableIOGet] = []
    expected_outputs: List[RunnableIOGet] = []
    code_excerpt: Optional[str] = None
    last_runs = []


class RunnableMetricsBucket(BaseModel):
    """
    Type of metadata of a specific bucket.

    start: start timestamp of the time range
    end: end timestamp of the time range
    run_count: count of runs part of this bucket
    success_rate: percentage of successful runs as float between 0 and 1
    average_runtime: average runtime of all runs in this bucket
    total_runtime: total runtime of all runs in this bucket
    """

    start: datetime
    end: datetime
    run_count: int
    success_rate: float
    average_runtime: int
    total_runtime: int


class RunnableMetricsGet(BaseModel):
    """
    Return type of the metrics endpoint.

    start: start timestamp of the time range
    end: end timestamp of the time range
    bucket_count: number of buckets time range was divided into
    metrics_buckets: list of metadata about each bucket
    overall_bucket: metadata for the whole time range
    preceding_bucket: metadata for the bucket directly preceding the time range
    """

    start: datetime
    end: datetime
    bucket_count: int
    metrics_buckets: List[RunnableMetricsBucket]
    overall_bucket: RunnableMetricsBucket
    preceding_bucket: RunnableMetricsBucket


class RunnableMetricsQuery(BaseModel):
    """
    Query parameters for RunnableMetricsGet.

    start: start timestamp of the time range
    end: end timestamp of the time range
    bucket_count: number of buckets to divide the time range into
    """

    start: datetime = Field(
        default_factory=lambda: datetime.utcnow() - timedelta(hours=24)
    )
    end: datetime = Field(default_factory=datetime.utcnow)
    bucket_count: int = 100


# NOTE QUESTION: do we use these classes?
class FunctionGet(RunnableGet):
    type: RunnableType = Field(RunnableType.function, const=True)


class FunctionGetDetailed(FunctionGet, RunnableGetDetailed):
    pass


class PipelineGet(RunnableGet):
    type: RunnableType = Field(RunnableType.pipeline, const=True)


class PipelineGetDetailed(PipelineGet, RunnableGetDetailed):
    pass
