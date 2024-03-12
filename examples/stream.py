from pipeline.cloud.pipelines import stream_pipeline

pipeline = "pipeline_id_or_pointer"
inputs = ["input string"]

print("Streaming pipeline:\n")
for result in stream_pipeline(pipeline, *inputs):
    print(result.outputs_formatted()[0], flush=True)
