import time
from datetime import datetime, timedelta

from pipeline.cloud import http
from pipeline.cloud.schemas.runs import ClusterRunResult, RunState


def get_run(run_id: str):
    response = http.get(f"/v4/runs/{run_id}")
    result = ClusterRunResult.parse_raw(response.text)
    return result


def poll_for_run_completion(run_id, timeout_secs: int = 5 * 60, interval_secs: int = 5):
    start_time = datetime.now()
    while datetime.now() - start_time < timedelta(seconds=timeout_secs):
        run = get_run(run_id)
        if RunState.is_terminal(run.state):
            return run
        print(f"Waiting for run {run_id} to finish...")
        time.sleep(interval_secs)
    raise TimeoutError(f"Run {run_id} did not finish in the allotted time")
