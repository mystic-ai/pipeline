from typing import List


from pipeline import Pipeline, pipeline_function, Variable
from pipeline.model import pipeline_model


@pipeline_model
class MyModel(object):
    def __init__(self):
        print("Created new ml model")

    @pipeline_function
    def predict(self, input_ids: List[int]) -> str:
        return "model inference result"


@pipeline_function
def tokenize_str(a: str) -> list:
    return [0, 0, 0, 0]


with Pipeline() as pipeline:
    input_str = Variable(variable_type=str, is_input=True)

    ml_model = MyModel()
    token_ids = tokenize_str(input_str)
    output_str = ml_model.predict(token_ids)

    pipeline.output(output_str)

output_pipeline = Pipeline.get_pipeline()

print("Pipeline graph:\n%s" % output_pipeline.json())
print(pipeline.run("Hello"))
