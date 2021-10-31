from typing import List, Tuple, Any


from pipeline import Pipeline, pipeline_function, Variable
from pipeline.model import pipeline_model


@pipeline_model(file_or_dir="/lol", compress_tar=False)
class MyModel(object):
    model = None

    def __init__(self):
        print("Created new ml model")

    @pipeline_function
    def predict(self, input_ids: List[int]) -> str:
        return str(input_ids)

    @pipeline_function
    def load(self, model_file) -> None:
        print("Loading model")
        self.model = None


@pipeline_function
def tokenize_str(
    a: str,
) -> list:  # This breaks if output -> List[int] instead of -> list
    output_list = list([ord(val) for val in a])
    return output_list


with Pipeline() as pipeline:
    input_str = Variable(variable_type=str, is_input=True)

    ml_model = MyModel()
    token_ids = tokenize_str(input_str)
    output_str = ml_model.predict(token_ids)

    pipeline.output(output_str)
    # pipeline.output(token_ids)

output_pipeline = Pipeline.get_pipeline()

# print("Pipeline graph:\n%s" % output_pipeline.json())
print(pipeline.run("Hello"))
