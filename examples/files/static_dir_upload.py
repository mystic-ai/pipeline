import tempfile
import time

from pipeline import Pipeline, Variable, pipe
from pipeline.cloud.compute_requirements import Accelerator
from pipeline.cloud.environments import create_environment
from pipeline.cloud.pipelines import run_pipeline, upload_pipeline
from pipeline.configuration import current_configuration
from pipeline.objects.graph import Directory

current_configuration.set_debug_mode(True)


@pipe
def test(i: int, static_directory: Directory) -> list:
    time.sleep(1)

    files_in_dir = []
    if static_directory.path is not None:
        files_in_dir = [str(path) for path in static_directory.path.glob("*")]

    return files_in_dir


tmp_dir = tempfile.TemporaryDirectory()

tmp_files = [tempfile.NamedTemporaryFile(dir=tmp_dir.name) for _ in range(5)]

with Pipeline() as builder:
    input_var = Variable(
        int,
        gt=0,
        lt=20,
    )

    local_file = Directory(path=tmp_dir.name)
    b = test(input_var, local_file)

    builder.output(b)

pl = builder.get_pipeline()


env_id = create_environment(
    name="basic2",
    python_requirements=[
        "numpy==1.25.0",
    ],
    allow_existing=True,
)


for i, tmp_file in enumerate(tmp_files):
    tmp_file.write(f"Hello world (i={i})".encode())
    tmp_file.flush()


result = upload_pipeline(
    pl,
    "debugging-pipeline:latest",
    environment_id_or_name="basic2",
    accelerators=[
        Accelerator.cpu,
    ],
)

[tf.close() for tf in tmp_files]
tmp_dir.cleanup()

output = run_pipeline(
    result.id,
    5,
)
print(output.outputs())
