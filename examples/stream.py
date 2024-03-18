from pipeline.cloud.pipelines import stream_pipeline

pipeline = "pipeline_id_or_pointer"
inputs = ["input string"]

print("Streaming pipeline:\n")
for result in stream_pipeline(pipeline, *inputs):
    if result.error:
        print(result)
    else:
        print(result.outputs_formatted()[0], flush=True)
