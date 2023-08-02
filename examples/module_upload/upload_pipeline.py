from custom_module.custom_model import MyModel

from pipeline import Pipeline, Variable, pipeline_function
from pipeline.cloud.environments import create_environment
from pipeline.cloud.pipelines import upload_pipeline


@pipeline_function
def get_number(var: int) -> int:
    model = MyModel()
    return model.random()


with Pipeline() as builder:
    var = Variable(int, is_input=True)

    builder.add_variables(var)

    res = get_number(var)

    builder.output(res)


pl = builder.get_pipeline()

env_name = "module_test_env"
try:
    create_environment(
        env_name,
        ["numpy"],
    )
except Exception:
    ...

print(
    upload_pipeline(
        pl,
        "ph/module_test:main",
        environment_id_or_name=env_name,
        minimum_cache_number=1,
        modules=[
            "custom_module",
        ],
    )
)
