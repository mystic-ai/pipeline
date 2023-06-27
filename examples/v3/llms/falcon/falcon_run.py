from pipeline.v3.pipelines import run_pipeline

pointer = "mystic/falcon-7b-instruct:standard"

inputs = [
    "Hey there, my name is Paul and I'm a software engineer.",
    {},
]

output = run_pipeline(
    pointer,
    *inputs,
)
print(output.result.result_array())
