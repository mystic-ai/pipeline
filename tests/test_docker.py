import os

from pipeline import Pipeline, Variable, docker, pipeline_function


def test_dockerfiles(tmp_path):
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

    test_dir = tmp_path / "docker_test"
    docker.create_pipeline_api([graph], output_dir=test_dir)

    # Check for the creation of docker files
    assert os.path.exists(f"{test_dir}/Dockerfile") and os.path.exists(
        f"{test_dir}/docker-compose.yml"
    )
    # Check for the serialization of the graph
    assert os.path.exists(f"{test_dir}/AddNumbers.graph")

    # TODO validate that the docker files are correct
