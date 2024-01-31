from pipeline.cloud import http
from pipeline.cloud.schemas.runs import ClusterRunResult


def get_run(run_id: str):
    response = http.get(f"/v4/runs/{run_id}")
    result = ClusterRunResult.parse_raw(response.text)
    return result
