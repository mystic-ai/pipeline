from pipeline import Pipeline, Variable, pipeline_function
from pipeline.objects.environment import Dependency, Environment

custom_env = Environment(
    environment_name="my-custom-env",
    dependencies=[Dependency(dependency_string="numpy==1.23.4")],
)


@pipeline_function
def add_numbers(a: float, b: float) -> float:
    return a + b


with Pipeline("custom-env-pipeline", environment=custom_env) as builder:
    a = Variable(float, is_input=True)
    b = Variable(float, is_input=True)
    builder.add_variables(a, b)

    c = add_numbers(a, b)

    builder.output(c)

my_pipeline = Pipeline.get_latest_pipeline()

res = my_pipeline.run(2.0, 4.0)

print(res)
