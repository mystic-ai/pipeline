from pipeline import PipelineCloud

api = PipelineCloud()
api.authenticate(project="Test project")

print(api.active_project)
