import datetime

from pipeline.cloud.pipelines import run_pipeline

start_time = datetime.datetime.now()

pointer = "pipeline_id"

result = run_pipeline(pointer, "input string")

end_time = datetime.datetime.now()

total_time = (end_time - start_time).total_seconds()

if result.error:
    print(f"Error: {result.error.json(indent=2)}")
else:
    print(
        f"Total time taken: {total_time}, run id: {result.id}, result: {result.outputs_formatted()}"
    )
