import datetime
import sys
import time

from pipeline.v3.pipelines import get_pipeline_run, run_pipeline
from pipeline.v3.schemas.runs import RunState

pointer = "ph/test:test"
print(f"Running test pipeline async: {pointer}")

start_time = datetime.datetime.now()

result = run_pipeline(
    pointer,
    1,
    async_run=True,
)


while not (
    result.state
    in [
        RunState.completed,
        RunState.failed,
        RunState.rate_limited,
        RunState.lost,
        RunState.no_environment_installed,
    ]
):
    result = get_pipeline_run(result.id)
    sys.stdout.write(
        f"\rWaiting for pipeline to complete... (state: {result.state.name})"
    )
    sys.stdout.flush()
    time.sleep(0.5)

print("")


end_time = datetime.datetime.now()

total_time = (end_time - start_time).total_seconds() * 1e3

print(
    "Total time taken: %.3f ms, result: '%s'"
    % (
        total_time,
        result.result.result_array(),
    )
)
