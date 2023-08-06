from pipeline import Pipeline, Variable, pipeline_function
from pipeline.cloud.compute_requirements import Accelerator
from pipeline.cloud.environments import create_environment
from pipeline.cloud.pipelines import upload_pipeline


@pipeline_function
def pi_sample(i: int) -> bool:
    import numpy as np

    x, y = np.random.rand(2)
    return bool(x**2 + y**2 < 1.0)


with Pipeline() as builder:
    input_var = Variable(
        int,
        gt=0,
        lt=100,
        title="",
    )

    b = pi_sample(input_var)

    builder.output(b)

pl = builder.get_pipeline()


env_id = create_environment(
    name="paulh/numpy",
    python_requirements=["numpy==1.24.3"],
    allow_existing=True,
)
print(f"New environment ID = {env_id}")
print(
    "Environment will be pre-emptively cached on compute resources so please "
    "wait a few mins before using..."
)

result = upload_pipeline(
    pl,
    "paulh/test",
    environment_id_or_name="paulh/numpy",
    minimum_cache_number=1,
    accelerators=[
        Accelerator.cpu,
    ],
)

pipeline_id = result.id
print(f"New pipeline ID = {pipeline_id}")
