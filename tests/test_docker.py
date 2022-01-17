import os

from pipeline import docker
from pipeline import Pipeline, Variable, pipeline_function


def test_dockerfiles():
    @pipeline_function
    def add_numbers(in_1: float, in_2: float) -> float:
        return in_1 + in_2

    with Pipeline("AddNumbers") as builder:
        in_1 = Variable(float, is_input=True)
        in_2 = Variable(float, is_input=True)
        builder.add_variables(in_1, in_2)

        sum = add_numbers(in_1, in_2)

        builder.output(sum)

    graph = Pipeline.get_pipeline("AddNumbers")
    output = graph.run(3.0, 4.0)
    assert output[0] == 3 + 4

    docker.create_pipeline_api([graph], output_dir="./")

    # Check for the creation of docker files
    assert os.path.exists("./Dockerfile") and os.path.exists("./docker-compose.yml")
    # Check for the serialization of the graph
    assert os.path.exists("./AddNumbers.graph")

    # TODO validate that the docker files are correct
