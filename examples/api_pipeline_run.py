from pipeline import Pipeline, PipelineCloud, Variable, pipeline_function

api = PipelineCloud()
api.authenticate()


@pipeline_function
def add_lol(a: str) -> str:
    return a + " lol"


with Pipeline("AddLol") as builder:
    str_1 = Variable(str, is_input=True)
    builder.add_variable(str_1)
    res_1 = add_lol(str_1)

    builder.output(res_1)

test_pipeline = Pipeline.get_pipeline("AddLol")
upload_output = api.upload_pipeline(test_pipeline)

print(api.run_pipeline(upload_output, "Hi I like to")["run_state"])
