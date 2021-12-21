from pipeline.objects import (
    Pipeline,
    pipeline_function,
    pipeline_model,
)


# Test basic Pipeline
def test_with_exit():
    @pipeline_model()
    class CustomModel:
        def __init__(self, model_path="", tokenizer_path=""):
            self.model_path = model_path
            self.tokenizer_path = tokenizer_path

        @pipeline_function
        def predict(self, input: str, **kwargs: dict) -> str:
            return input + " lol"

        @pipeline_function
        def load(self, path: str) -> None:
            print("load")

    with Pipeline("test") as my_pipeline:
        in_1 = my_pipeline.add_variable(str, is_input=True)
        my_model = CustomModel()
        my_model.predict(in_1)
        my_pipeline.output(in_1)

    output = Pipeline.run("test", "hey")
    assert output == ["hey lol"]
