from httpx import HTTPStatusError

from pipeline import Pipeline, Variable, pipeline_function
from pipeline.v3.compute_requirements import Accelerator
from pipeline.v3.environments import create_environment
from pipeline.v3.pipelines import upload_pipeline


@pipeline_function
def pi_sample(i: int) -> bool:
    import numpy as np

    x, y = np.random.rand(2)
    return bool(x**2 + y**2 < 1.0)


with Pipeline() as builder:
    input_var = Variable(int, is_input=True)

    builder.add_variables(input_var)
    b = pi_sample(input_var)

    builder.output(b)

pl = builder.get_pipeline()


try:
    env_id = create_environment(
        name="paulh/numpy", python_requirements=["numpy==1.24.3"]
    )
    print(f"New environment ID = {env_id}")
    print(
        "Environment will be pre-emptively cached on compute resources so please "
        "wait a few mins before using..."
    )
except HTTPStatusError as e:
    if e.response.status_code == 400:
        print("Environment already exists, skipping creation...")
        env_id = "numpy"

result = upload_pipeline(
    pl,
    "paulh/test",
    environment_id_or_name="environment_effdf578d81a4f298ffbab6c656a8229",
    minimum_cache_number=1,
    required_gpu_vram_mb=None,
    accelerators=[
        Accelerator.cpu,
    ],
)

pipeline_id = result.id
print(f"New pipeline ID = {pipeline_id}")
