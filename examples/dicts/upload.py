from pydantic import BaseModel, Field

from pipeline import Pipeline, Variable, pipeline_function
from pipeline.cloud.compute_requirements import Accelerator
from pipeline.cloud.environments import create_environment
from pipeline.cloud.pipelines import upload_pipeline


class MyKwargs(BaseModel):
    a: int = Field(10, lt=20, gte=1)
    b: str = Field("hello", min_length=2, max_length=10)


@pipeline_function
def do_something(kwargs: dict) -> str:
    return str(kwargs)


with Pipeline() as builder:
    input_var = Variable(
        dict,
        dict_schema=MyKwargs,
    )

    o = do_something(input_var)

    builder.output(o)

pl = builder.get_pipeline()


env_id = create_environment(
    name="numpy",
    python_requirements=["numpy==1.24.3"],
    allow_existing=True,
)

print(f"New environment ID = {env_id}")

result = upload_pipeline(
    pl,
    "paulh/dict-test:test",
    environment_id_or_name="numpy",
    minimum_cache_number=1,
    required_gpu_vram_mb=None,
    accelerators=[
        Accelerator.cpu,
    ],
)


pipeline_id = result.id
print(f"New pipeline ID = {pipeline_id}")
