from pipeline.cloud.pipelines import run_pipeline
from pipeline.cloud.runs import poll_for_run_completion
from pipeline.cloud.schemas.runs import RunState

pointer = "my_pipeline_id"
input_data = ["my input string"]

initial_result = run_pipeline(pointer, *input_data, async_run=True)
run_id = initial_result.id

result = poll_for_run_completion(run_id, timeout_secs=5 * 60, interval_secs=10)

if result.state != RunState.completed:
    print(f"Run id: {result.id}, state: {result.state}")
    if result.error:
        print(f"Error: {result.error.json(indent=2)}")
else:
    print(f"Run id: {result.id}, result: {result.outputs_formatted()}")
