import os
from dotenv import load_dotenv

load_dotenv("../hidden.env")

from pipeline.api import authenticate
from pipeline.api.pipeline import upload_pipeline
from pipeline.api.run import run_pipeline


from pipeline.objects.pipeline import Pipeline
from pipeline.objects.variable import Variable
from pipeline.objects import pipeline_function


api_token = os.getenv("TOKEN")
authenticate(api_token)


@pipeline_function
def add_lol(a: str) -> str:
    return a + " lol"


with Pipeline("AddLol") as builder:
    str_1 = Variable(str, is_input=True)

    res_1 = add_lol(str_1)

    builder.output(res_1)

test_pipeline = Pipeline.get_pipeline("AddLol")
upload_output = upload_pipeline(test_pipeline)
print("Uploaded pipeline response:%s" % upload_output.dict())

print(run_pipeline(upload_output, "Hi I like to"))
