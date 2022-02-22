import json


from pipeline import Pipeline, PipelineCloud, Variable, pipeline_function

api = PipelineCloud()
api.authenticate()


@pipeline_function
def add_lol(a: str, b: float) -> str:
    return a + " lol"


upload_output = api.upload_function(add_lol.__function__.__pipeline_function__)
print(upload_output)
