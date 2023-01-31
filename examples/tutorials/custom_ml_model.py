from typing import List

from pipeline import Pipeline, Variable, pipeline_function, pipeline_model


@pipeline_model
class MyModel:
    model = None

    def __init__(self):
        print("Created new ml model")

    @pipeline_function
    def predict(self, input_ids: List[int]) -> str:
        return str(input_ids)

    @pipeline_function
    def load(self, model_file=None) -> None:
        print("Loading model")
        self.model = None


@pipeline_function
def tokenize_str(
    a: str,
) -> list:  # This breaks if output -> List[int] instead of -> list
    output_list = list([ord(val) for val in a])
    return output_list


with Pipeline("ml-pipeline") as pipeline:
    input_str = Variable(type_class=str, is_input=True)
    pipeline.add_variable(input_str)

    ml_model = MyModel()
    token_ids = tokenize_str(input_str)
    output_str = ml_model.predict(token_ids)

    pipeline.output(output_str)

output_pipeline = Pipeline.get_pipeline("ml-pipeline")

print(output_pipeline.run("Hello"))

output_pipeline.save("./examples/output_pipeline")
