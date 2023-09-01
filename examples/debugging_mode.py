from pipeline import Pipeline, Variable, pipe
from pipeline.cloud.compute_requirements import Accelerator
from pipeline.cloud.environments import create_environment
from pipeline.cloud.pipelines import run_pipeline, upload_pipeline
from pipeline.configuration import current_configuration

current_configuration.set_debug_mode(True)


# Make local pipeline
@pipe
def pi_sample(i: int) -> str:
    import time

    for i in range(i):
        print(f"We're in the loop (count={i})")
        time.sleep(0.5)

    return "done"


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

# Create remote environment


create_environment(
    name="debugging-test-env-23",
    python_requirements=["numpy==3.59.11"],
)

# # Upload pipeline
remote_pipeline = upload_pipeline(
    pl,
    "debugging-test-pipeline",
    "debugging-test-env",
    accelerators=[
        Accelerator.cpu,
    ],
)


output = run_pipeline("test:v5", 5)
