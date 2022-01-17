import os

from pipeline import docker
from pipeline import Pipeline, Variable, pipeline_function


def test_dockerfiles():
    @pipeline_function
    def add_numbers(input_dict: dict) -> float:
        return input_dict["in_1"] + input_dict["in_2"]

    with Pipeline("AddNumbers") as builder:
        in_1 = Variable(dict, is_input=True)
        builder.add_variables(in_1)

        sum = add_numbers(in_1)

        builder.output(sum)

    graph = Pipeline.get_pipeline("AddNumbers")
    output = graph.run(3, 4)
    assert output[0] == 3 + 4

    docker.create_pipeline_api([graph], output_dir="./")

    # Check for the creation of docker files
    assert os.path.exists("./Dockerfile") and os.path.exists("./docker-compose.yml")
    # Check for the serialization of the graph
    assert os.path.exists("./AddNumbers.graph")

    # TODO validate that the docker files are correct
