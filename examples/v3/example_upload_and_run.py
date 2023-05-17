import datetime

from pipeline import Pipeline, Variable, pipeline_function
from pipeline.v3 import create_environment, run_pipeline, upload_pipeline


@pipeline_function
def pi_sample(i: int) -> bool:
    import numpy as np

    x, y = np.random.rand(2)
    return bool(x**2 + y**2 < 1.0)


with Pipeline("pi-approx") as builder:
    input_var = Variable(int, is_input=True)
    builder.add_variables(input_var)
    b = pi_sample(input_var)
    builder.output(b)

pl = Pipeline.get_pipeline("pi-approx")


env_id = create_environment(name="numpy", python_requirements=["numpy==1.24.3"])
print(f"New environment ID = {env_id}")

result = upload_pipeline(pl, environment_id_or_name=env_id)
pipeline_id = result.json()["id"]
print(f"New pipeline ID = {pipeline_id}")

start_time = datetime.datetime.now()

result = run_pipeline(pipeline_id, 1)

end_time = datetime.datetime.now()

total_time = (end_time - start_time).total_seconds() * 1e3

print("Total time taken: %.3f ms, result: '%s'" % (total_time, result))
