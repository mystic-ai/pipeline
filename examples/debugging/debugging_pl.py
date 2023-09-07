import time

from pipeline import Pipeline, Variable, pipe
from pipeline.cloud.compute_requirements import Accelerator
from pipeline.cloud.environments import create_environment
from pipeline.cloud.pipelines import run_pipeline, upload_pipeline
from pipeline.configuration import current_configuration

current_configuration.set_debug_mode(True)


@pipe
def test(i: int) -> str:
    for i in range(i):
        print(f"Pos:{i}")
        time.sleep(0.5)
    print("PRINT ME")
    return "Done"


with Pipeline() as builder:
    input_var = Variable(
        int,
        gt=0,
        lt=20,
    )

    b = test(input_var)

    builder.output(b)

pl = builder.get_pipeline()


env_id = create_environment(
    name="basic4",
    python_requirements=["pandas==2.0.1"],
    allow_existing=True,
)

result = upload_pipeline(
    pl,
    "debugging-pipeline:latest",
    environment_id_or_name="basic4",
    minimum_cache_number=1,
    accelerators=[
        Accelerator.cpu,
    ],
)

output = run_pipeline("debugging-pipeline:latest", 5)
