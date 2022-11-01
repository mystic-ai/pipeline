from unicodedata import name
from pipeline import Pipeline, Variable, pipeline_function, Interface
from pydantic import BaseModel


class InferenceParams(BaseModel):
    num_repeats: int
    num_fake: int


demo_inf_params: InferenceParams = {"num_repeats": 20, "num_fake": 48}


@pipeline_function
def math(a: float, num_repeats: float = 5) -> list[float]:
    return [a] * num_repeats


with Pipeline("Interface Demo") as pipeline:
    float_1 = Variable(type_class=float, is_input=True, name="base_float")
    arguments = Interface(
        type_class=InferenceParams, is_input=True, name="inference_params"
    )
    pipeline.add_variables(float_1, arguments)

    output_1 = math(float_1, arguments.grab("num_samples"))

    pipeline.output(output_1)

output_pipeline = Pipeline.get_pipeline("Interface Demo")
print(output_pipeline.run(5.0, InferenceParams.parse_obj(demo_inf_params)))
