import datetime

from pipeline.cloud.pipelines import run_pipeline

start_time = datetime.datetime.now()

pointer = "pipeline_5b74791689c9451e82f856cc2960f8a7"

result = run_pipeline(
    pointer, "mountains", {"denoising_end": 0.8, "num_inference_steps": 25}
)

end_time = datetime.datetime.now()

total_time = (end_time - start_time).total_seconds()

print(result)
print(f"Total time taken: {total_time}, result: {result.outputs_formatted()}")
