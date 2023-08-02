from pipeline.cloud.environments import create_environment

env_id = create_environment(name="numpy", python_requirements=["numpy==1.24.3"])
print(f"New environment ID = {env_id}")
print(
    "Environment will be pre-emptively cached on compute resources so please "
    "wait a few mins before using..."
)
