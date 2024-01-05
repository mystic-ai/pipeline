import datetime

from pipeline.cloud.pipelines import run_pipeline

start_time = datetime.datetime.now()

pointer = "betauser/simple:v1"

result = run_pipeline(pointer, "a")

end_time = datetime.datetime.now()

total_time = (end_time - start_time).total_seconds()

print(result.id)
print(f"Total time taken: {total_time}, result: {result.outputs_formatted()}")
print(f"Total time taken: {total_time}, result: {result.error}")
