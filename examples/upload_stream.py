from pipeline import Pipeline, pipe
from pipeline.cloud.environments import create_environment
from pipeline.cloud.pipelines import upload_pipeline
from pipeline.objects.graph import Stream, Variable


@pipe
def streaming_function(input_str: str) -> Stream[str]:
    import time

    for i in range(10):
        time.sleep(0.5)
        yield str(i)


with Pipeline() as builder:
    input_str = Variable(type_class=str)
    input_dict = Variable(type_class=dict)

    output_str = streaming_function(input_str)

    builder.output(output_str)

pl = builder.get_pipeline()

env_name = "streaming-test"
try:
    env_id = create_environment(
        name=env_name,
        python_requirements=[
            "numpy",
        ],
    )
    print(f"New environment ID = {env_id}")
    print(
        "Environment will be pre-emptively cached on compute resources so please "
        "wait a few mins before using..."
    )
except Exception:
    print("Environment already exists, using existing environment...")

new_pipeline = upload_pipeline(
    pl,
    name="ph/stream-test:test",
    environment_id_or_name=env_name,
    minimum_cache_number=1,
)

print(f"Uploaded pipeline: {new_pipeline.id}")
