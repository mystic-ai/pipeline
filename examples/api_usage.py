from dotenv import load_dotenv

from pipeline import Pipeline, PipelineCloud, Variable, pipeline_function

#####################################
# set env vars
#####################################
load_dotenv("hidden.env")


#####################################
# setup standalone api and auth
#####################################
api = PipelineCloud()
api.authenticate()


#####################################
# set pipeline function
#####################################
@pipeline_function
def multiply(a: float, b: float) -> float:
    return a * b


#####################################
# use ctx manager to configure pipeline
#####################################
with Pipeline("MathsTest") as builder:
    flt_1 = Variable(variable_type=float, is_input=True)
    flt_2 = Variable(variable_type=float, is_input=True)

    res_1 = multiply(flt_1, flt_2)

    builder.output(res_1)


#####################################
# Upload Pipeline with built in API
#####################################
remote_schema = Pipeline.upload("MathsTest")


#####################################
# Upload Pipeline with standalone API
#####################################
test_pipeline = Pipeline.get_pipeline("MathsTest")
upload_output = api.upload_pipeline(test_pipeline)
