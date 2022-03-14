import json
from pipeline import Pipeline, PipelineCloud, Variable, pipeline_function

api = PipelineCloud()
api.authenticate(project="Test project")

print(api.active_project)
