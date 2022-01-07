import os

from pipeline import Pipeline, Variable, pipeline_function
from pipeline.api import authenticate
from pipeline.api.pipeline import upload_pipeline
from pipeline.api.run import run_pipeline

api_token = os.getenv("TOKEN")
authenticate(api_token)


@pipeline_function
def add_lol(a: str) -> str:
    return a + " lol"


with Pipeline("AddLol") as builder:
    str_1 = Variable(str, is_input=True)
    builder.add_variable(str_1)
    res_1 = add_lol(str_1)

    builder.output(res_1)

test_pipeline = Pipeline.get_pipeline("AddLol")
upload_output = upload_pipeline(test_pipeline)

print(run_pipeline(upload_output, "Hi I like to")["run_state"])
