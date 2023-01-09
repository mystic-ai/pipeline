from pipeline import Pipeline, Variable, pipeline_function
from pipeline.objects.environment import EnvironmentSession
from pipeline.objects.environment.premade import worker_torch_environment

worker_torch_environment.initialize(overwrite=True)


@pipeline_function
def add_numbers(a: float, b: float) -> float:
    return a + b


with Pipeline("custom-env-pipeline", environment=worker_torch_environment) as builder:
    a = Variable(float, is_input=True)
    b = Variable(float, is_input=True)
    builder.add_variables(a, b)

    c = add_numbers(a, b)

    builder.output(c)

my_pipeline = Pipeline.get_latest_pipeline()


with EnvironmentSession(environment=worker_torch_environment) as session:
    session.add_pipeline(my_pipeline)
    print(session.run_pipeline(my_pipeline, [1.0, 1.0]))

print("Done")
