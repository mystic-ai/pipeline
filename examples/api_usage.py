import os
from dotenv import load_dotenv

import pipeline.api
from pipeline import Pipeline, Variable, pipeline_function

load_dotenv("hidden.env")

api_token = os.getenv("TOKEN")
pipeline.api.authenticate(api_token, url="http://localhost:5001/")


@pipeline_function
def multiply(a: float, b: float) -> float:
    return a * b


with Pipeline("MathsTest") as builder:
    flt_1 = Variable(variable_type=float, is_input=True)
    flt_2 = Variable(variable_type=float, is_input=True)

    res_1 = multiply(flt_1, flt_2)

    builder.output(res_1)

test_pipeline = Pipeline.get_pipeline("MathsTest")
upload_output = pipeline.api.upload(multiply)
print(upload_output)
