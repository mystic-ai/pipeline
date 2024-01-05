import datetime

from pipeline.cloud.pipelines import run_pipeline

start_time = datetime.datetime.now()

pointer = "test:v1"

result = run_pipeline(pointer, 1)

end_time = datetime.datetime.now()

total_time = (end_time - start_time).total_seconds()

if result.error:
    print(f"Error: {result.error.json()}")
else:
    print(
        f"Total time taken: {total_time}\n"
        f"Run ID: {result.id}\n"
        f"Result: {result.outputs_formatted()}"
    )
