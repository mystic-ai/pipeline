from pipeline import PipelineCloud, pipeline_function

api = PipelineCloud()


@pipeline_function
def add_lol(a: str) -> str:
    return a + " lol"


upload_output = api.upload_function(add_lol.__function__.__pipeline_function__)
print(upload_output)
