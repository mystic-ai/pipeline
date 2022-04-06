from datetime import datetime, timedelta
from .base import BaseModel
from pydantic import Field
from typing import List


class MetricsBucket(BaseModel):
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
    metrics_buckets: List[MetricsBucket]
    overall_bucket: MetricsBucket
    preceding_bucket: MetricsBucket


class MetricsQuery(BaseModel):
    """
    Query parameters for generic metrics requests.

    start: start timestamp of the time range
    end: end timestamp of the time range
    bucket_count: number of buckets to divide the time range into
    """

    start: datetime = Field(
        default_factory=lambda: datetime.utcnow() - timedelta(hours=24)
    )
    end: datetime = Field(default_factory=datetime.utcnow)
    bucket_count: int = 100
