from pipeline.v3 import create_environment

env_id = create_environment(name="pandas", python_requirements=["pandas==2.0.0"])
print(f"New environment ID = {env_id}")
