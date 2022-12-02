from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional

from pydantic import Field

from .base import BaseModel


class DurationUnit(Enum):
    seconds = "seconds"
    minutes = "minutes"
    hours = "hours"
    days = "days"
    weeks = "weeks"


class Duration(BaseModel):
    """Representation of time duration in a given set of units.
    Used for specifiying bucket interval sizes, typically in metrics charts."""

    unit: DurationUnit
    value: int


class RunMetric(BaseModel):
    """
    Type of run metric data specified for a pipeline.
    """

    #: total number of runs executed
    run_count: int
    #: total number of failed runs
    failed_run_count: int
    #: total number of succeeded runs
    succeeded_run_count: int
    #: total run compute time (typically of succeeded runs)
    total_compute_ms: int


class RunHardwareMetric(BaseModel):
    resource_type: str
    total_percentage: float
    run_count: int
    average_runtime: int


class ProjectHardwareMetric(BaseModel):
    project_id: str
    project_name: str
    project_usage: List[RunHardwareMetric]


class HardwareMetric(BaseModel):
    start: datetime
    end: datetime
    projects: List[ProjectHardwareMetric]
    account_usage: List[RunHardwareMetric]


class RunMetricsBucket(BaseModel):
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


class RunMetricsGet(BaseModel):
    """
    Return type of the metrics endpoint on run related requests.

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
    metrics_buckets: List[RunMetricsBucket]
    overall_bucket: RunMetricsBucket
    preceding_bucket: RunMetricsBucket


class PipelineMetricsGet(RunMetricsGet):
    """Run metrics supplemented with pipeline meta data. Used in particular
    for list/paginated metrics views where it is important for the client to know which
    pipeline is which"""

    pipeline_id: str
    pipeline_name: str


class PipelineMetricsGetSummary(RunMetric):
    """Pipeline run summary metrics supplemented with pipeline meta data.
    Pipeline meta data is useful for list/paginated metrics views so the client
    can keep track of which pipeline is which"""

    pipeline_id: str
    pipeline_name: str


class MetricsQuery(BaseModel):
    """
    Query parameters for generic metrics requests.

    start: start timestamp of the time range
    end: end timestamp of the time range
    """

    start: datetime = Field(
        default_factory=lambda: datetime.utcnow() - timedelta(hours=24)
    )
    end: datetime = Field(default_factory=datetime.utcnow)


class RunMetricsQuery(MetricsQuery):
    """
    Query parameters for run metrics requests.

    start: start timestamp of the time range
    end: end timestamp of the time range
    bucket_count: number of buckets to divide the time range into
    """

    bucket_count: int = 100


class MetricsBucketsIntervalQuery(MetricsQuery):
    """
    Alternate query parameters for run metrics requests.
    Specify a bucket time interval/duration instead of a number of buckets
    """

    interval: Duration


class PipelineComputeGet(BaseModel):
    """
    Pipeline compute metrics for retrieving the number of completed runs
    and total compute time of all the runs on a pipeline
    """

    start: datetime
    end: datetime
    pipeline_id: str
    pipeline_name: str
    completed_run_count: int
    total_compute_ms: int


class TotalComputeGet(RunMetric):
    """Total compute time of all a users' runs"""

    start: datetime
    end: datetime
    #: change rate of run count
    run_count_rate: Optional[float]
    #: change rate of succeeded run count
    succeeded_run_count_rate: Optional[float]
    #: change rate of failed run count
    failed_run_count_rate: Optional[float]
    #: change rate of compute run time
    total_compute_ms_rate: Optional[float]


class PipelineRunMetricsData(BaseModel):
    """
    Overall and per-timestamp run metrics data for a given pipeline.
    Used as list item in return data for run metrics over multiple pipelines
    """

    pipeline_id: str
    pipeline_name: str
    overall_bucket: RunMetric
    metrics_buckets: List[RunMetric]


class PipelinesRunMetricsGet(MetricsQuery):
    """
    Response schema for retrieving run metrics (overall + per-timestamp)
    on a list of pipelines.
    """

    timestamps: List[datetime]
    data: List[PipelineRunMetricsData]
