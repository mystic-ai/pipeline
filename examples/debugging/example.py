import time

from tqdm import tqdm

from pipeline import Pipeline, Variable, pipe
from pipeline.cloud.compute_requirements import Accelerator
from pipeline.cloud.environments import create_environment
from pipeline.cloud.pipelines import run_pipeline, upload_pipeline
from pipeline.configuration import current_configuration

current_configuration.set_debug_mode(True)


@pipe
def test(i: int) -> str:
    for i in tqdm(range(i)):
        time.sleep(0.5)
    print("I'm done now, goodbye!")
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
    name="basic",
    python_requirements=[
        "tqdm",
    ],
    allow_existing=True,
)

result = upload_pipeline(
    pl,
    "debugging-pipeline:latest",
    environment_id_or_name="basic",
    accelerators=[
        Accelerator.cpu,
    ],
)

output = run_pipeline(
    result.id,
    5,
)
