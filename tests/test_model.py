from pipeline.objects import Pipeline, Variable, pipeline_function, pipeline_model


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
        def load(self) -> None:
            print("load")

    with Pipeline("test") as my_pipeline:
        in_1 = Variable(str, is_input=True)
        my_pipeline.add_variable(in_1)

        my_model = CustomModel()
        str_1 = my_model.predict(in_1)

        my_pipeline.output(str_1)

    test_pipeline = Pipeline.get_pipeline("test")
    output = test_pipeline.run("test", "hey")

    assert output == ["hey lol"]
