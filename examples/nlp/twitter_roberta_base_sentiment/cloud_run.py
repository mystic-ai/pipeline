import datetime

from pipeline.cloud.pipelines import run_pipeline
from pipeline.cloud.schemas.runs import Run
from pipeline.configuration import current_configuration

from .upload import PIPELINE_POINTER

current_configuration.set_debug_mode(True)

print(f"Running pipeline... {PIPELINE_POINTER})")

start_time = datetime.datetime.now()

result: Run = run_pipeline(PIPELINE_POINTER, "I don't know how I feel tbh.")

end_time = datetime.datetime.now()

total_time = (end_time - start_time).total_seconds() * 1e3

print(
    "Total time taken: %.3f ms, result: '%s'"
    % (total_time, result.result.result_array())
)
