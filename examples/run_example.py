import datetime

from pipeline.cloud.pipelines import run_pipeline

start_time = datetime.datetime.now()

pointer = "paulh/test:v1"

result = run_pipeline(pointer, 1)

end_time = datetime.datetime.now()

total_time = (end_time - start_time).total_seconds()

print(f"Total time taken: {total_time}, result: {result.result.result_array()}")
