from pipeline import Pipeline, PipelineCloud, Variable, pipeline_function

#####################################
# setup API client
#####################################
# Connect to pipeline cloud, you can set your token in one of three ways:
# 1. Pass it in as a kwarg: PipelineCloud(token=...)
# 2. Use pipeline CLI: pipeline login -t <token>
# 3. Set the PIPELINE_API_TOKEN=... environment variable
api = PipelineCloud()


#####################################
# set pipeline function
#####################################
@pipeline_function
def multiply(a: float, b: float) -> float:
    return a * b


#####################################
# use ctx manager to configure pipeline
#####################################
with Pipeline("MathsTest") as pipeline:
    flt_1 = Variable(type_class=float, is_input=True)
    flt_2 = Variable(type_class=float, is_input=True)
    pipeline.add_variables(flt_1, flt_2)

    res_1 = multiply(flt_1, flt_2)

    pipeline.output(res_1)


#####################################
# Upload Pipeline
#####################################
output_pipeline = Pipeline.get_pipeline("MathsTest")
uploaded_pipeline = api.upload_pipeline(output_pipeline)


#####################################
# Run Pipeline
#####################################
run_result = api.run_pipeline(uploaded_pipeline, [5.0, 6.0])

#####################################
# Get result
#####################################
try:
    result_preview = run_result.result_preview
except KeyError:
    result_preview = "unavailable"
print("Run result:", result_preview)
