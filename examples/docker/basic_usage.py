from pipeline import Pipeline, Variable, docker, pipeline_function


@pipeline_function
def add_numbers(in_1: float, in_2: float) -> float:
    return in_1 + in_2


with Pipeline("AddNumbers") as builder:
    in_1 = Variable(float, is_input=True)
    in_2 = Variable(float, is_input=True)
    builder.add_variables(in_1, in_2)

    sum = add_numbers(in_1, in_2)

    builder.output(sum)


docker.create_pipeline_api([Pipeline.get_pipeline("AddNumbers")])
