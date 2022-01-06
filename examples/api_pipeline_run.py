from dotenv import load_dotenv

from pipeline import Pipeline, PipelineApi, Variable, pipeline_function

load_dotenv("../hidden.env")


api = PipelineApi()
api.authenticate()


@pipeline_function
def add_lol(a: str) -> str:
    return a + " lol"


with Pipeline("AddLol") as builder:
    str_1 = Variable(str, is_input=True)

    res_1 = add_lol(str_1)

    builder.output(res_1)

test_pipeline = Pipeline.get_pipeline("AddLol")
upload_output = api.upload_pipeline(test_pipeline)

print(api.run_pipeline(upload_output, "Hi I like to")["run_state"])
