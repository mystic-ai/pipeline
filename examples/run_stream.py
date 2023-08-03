from pipeline.cloud.pipelines import stream_pipeline

pointer = "ph/stream-test:test"

inputs = ["test input", dict()]

for output in stream_pipeline(pointer, *inputs):
    print(output)

print("Done!")
