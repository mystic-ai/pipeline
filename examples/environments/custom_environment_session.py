from pipeline import Pipeline, Variable, pipeline_function
from pipeline.objects.environment import Environment, EnvironmentSession

custom_env = Environment(
    environment_name="my-custom-env",
    dependencies=["numpy==1.23.4"],
)

custom_env.add_dependencies(["dill", "pipeline-ai"])
custom_env.add_dependencies("pipeline-ai")
exit()
custom_env.initialize(overwrite=True)


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


with EnvironmentSession(environment=custom_env) as session:
    session.add_pipeline(my_pipeline)
    print(session.run_pipeline(my_pipeline, [1.0, 1.0]))

print("Done")
