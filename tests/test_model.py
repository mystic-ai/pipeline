from pipeline.objects import Pipeline, Variable, pipeline_function, PipelineBase
from pipeline.objects.decorators import PipelineBase


# Test basic Pipeline
def test_with_exit():
    class CustomModel(PipelineBase):
        def __init__(self, model_path="", tokenizer_path="", file_or_dir: str = None, compress_tar=False):
            super().__init__(file_or_dir, compress_tar)
            
            self.model_path = model_path
            self.tokenizer_path = tokenizer_path

        @pipeline_function
        def predict(self, input: str, **kwargs: dict) -> str:
            return input + " lol"

        @pipeline_function
        def load(self) -> None:
            print("load")

    with Pipeline("test") as my_pipeline:
        in_1 = Variable(str, is_input=True)
        my_pipeline.add_variable(in_1)

        my_model = CustomModel()
        str_1 = my_model.predict(in_1)

        my_pipeline.output(str_1)

    output = Pipeline.run("test", "hey")
    assert output == ["hey lol"]
