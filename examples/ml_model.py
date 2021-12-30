from typing import List

from pipeline import PipelineV2


class MyModel(object):
    model = None

    def __init__(self):
        print("Created new ml model")

    def predict(self, input_ids: List[int]) -> str:
        return str(input_ids)

    def load(self, model_file) -> None:
        print("Loading model")
        self.model = None


def tokenize_str(
    a: str,
) -> list:  # This breaks if output -> List[int] instead of -> list
    output_list = list([ord(val) for val in a])
    return output_list


pipeline = PipelineV2("ML_pipeline")
ml_model = MyModel()
pipeline.model(ml_model)
pipeline.set_stages(tokenize_str, MyModel.predict)

output = pipeline.run("Hello")

print(output)
pipeline.save("/examples")
