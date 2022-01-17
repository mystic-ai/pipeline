from pipeline import docker
from pipeline import Pipeline, Variable, pipeline_function


@pipeline_function
def add_numbers(input_dict: dict) -> float:
    return input_dict["in_1"] + input_dict["in_2"]


with Pipeline("AddNumbers") as builder:
    in_1 = Variable(dict, is_input=True)
    builder.add_variables(in_1)

    sum = add_numbers(in_1)

    builder.output(sum)


docker.create_pipeline_api([Pipeline.get_pipeline("AddNumbers")])
