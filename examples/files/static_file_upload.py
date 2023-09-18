import time

from pipeline import Pipeline, Variable, pipe
from pipeline.cloud.compute_requirements import Accelerator
from pipeline.cloud.environments import create_environment
from pipeline.cloud.pipelines import run_pipeline, upload_pipeline
from pipeline.configuration import current_configuration
from pipeline.objects.graph import File

current_configuration.set_debug_mode(True)


@pipe
def test(i: int, static_file: File) -> str:
    time.sleep()

    print(f"Got file with path content: {static_file.path.read_text()}")

    return "Done"


with Pipeline() as builder:
    input_var = Variable(
        int,
        gt=0,
        lt=20,
    )

    local_file = File.from_object("I AM JUST A STRING")
    b = test(input_var, local_file)

    builder.output(b)

pl = builder.get_pipeline()


env_id = create_environment(
    name="basic",
    python_requirements=[
        "tqdm==4.65.0",
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
