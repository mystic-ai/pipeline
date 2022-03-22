import pip
from pipeline import Pipeline, pipeline_function, Variable

from pipeline.tags import ComputeTags, ConditionalArgs


@pipeline_function(
    tags=[
        ComputeTags.VRAM32GB(
            cond=lambda ConditionalArgs: ConditionalArgs.inputs[1].get(
                "output_length", default=0
            )
            > 20
        )
    ]
)
def calculate_something(input_str: str, conds: dict) -> str:
    return input_str + " lol"


with Pipeline(new_pipeline_name="Example") as builder:
    input_str = Variable(str, is_input=True)
    input_conds = Variable(dict, is_input=True)

    builder.add_variables(input_str, input_conds)

    output_str = calculate_something(input_str, input_conds)

    builder.output(output_str)


api.upload_pipeline(my_pipeline)
