import time
from datetime import datetime, timedelta

from pipeline.cloud.pipelines import run_pipeline
from pipeline.cloud.runs import get_run
from pipeline.cloud.schemas.runs import RunState


def poll_for_run_completion(run_id, timeout_secs: int = 5 * 60, interval_secs: int = 5):
    start_time = datetime.now()
    while datetime.now() - start_time < timedelta(seconds=timeout_secs):
        run = get_run(run_id)
        if RunState.is_terminal(run.state):
            return run
        print(f"Waiting for run {run_id} to finish...")
        time.sleep(interval_secs)
    raise TimeoutError(f"Run {run_id} did not finish in the allotted time")


pointer = "pipeline_id"

initial_result = run_pipeline(pointer, "an input string", async_run=True)
run_id = initial_result.id

result = poll_for_run_completion(run_id, interval_secs=1)

if result.state != RunState.completed:
    print(f"Run id: {result.id}, state: {result.state}")
    if result.error:
        print(f"Error: {result.error.json(indent=2)}")
else:
    print(f"Run id: {result.id}, result: {result.outputs_formatted()}")
