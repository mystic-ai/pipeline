import datetime

from pipeline.cloud.pipelines import run_pipeline
from pipeline.cloud.schemas.runs import Run

pipeline_id = "mystic/neo:main"
print(f"Running GPTNeo pipeline... (pipeline_id: {pipeline_id})")

start_time = datetime.datetime.now()

result: Run = run_pipeline(pipeline_id, "Hello, my name is", 50)


end_time = datetime.datetime.now()

total_time = (end_time - start_time).total_seconds() * 1e3

print(
    "Total time taken: %.3f ms, result: '%s'"
    % (total_time, result.result.result_array())
)
