import time

from pipeline import Pipeline, Variable, pipeline_function
from pipeline.objects.environment import Dependency, Environment, EnvironmentSession

custom_env = Environment(
    environment_name="my-custom-env",
    dependencies=[Dependency(dependency_string="numpy==1.23.4")],
)

custom_env.initialize(overwrite=False)


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


# import subprocess

# with subprocess.Popen(
#     [
#         "python",
#         "-c",
#         "import time\ntime.sleep(3)\nprint(555)\ntime.sleep(10)\nprint(123,flush=True)",
#     ],
#     stdout=subprocess.PIPE,
#     stderr=subprocess.PIPE,
# ) as proc:
#     print(proc.stdout.readline())
# exit()


with EnvironmentSession(environment=custom_env) as session:
    time.sleep(2)
    for i in range(5):
        print(f"Attempt: {i}")
        # print(f"Alive={session._proc.communicate(timeout=1)}")
        print(f"out={session._proc.stdout.read()}")
        time.sleep(1)
