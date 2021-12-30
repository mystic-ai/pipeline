from pipeline import PipelineV2


def test_model():
    p = PipelineV2("test")

    class CustomModel:
        def __init__(self, model_path="", tokenizer_path=""):
            self.model_path = model_path
            self.tokenizer_path = tokenizer_path

        @p.stage
        def predict(self, input: str, **kwargs: dict) -> str:
            return input + " lol"

        def load(self, path: str) -> None:
            print("load")

    p.model(CustomModel())
    output = p.run("hey")
    results = p.get_results()
    assert results == ["hey lol"]
    assert output == p.get_named_results()


# Test basic Pipeline
# def test_with_exit():
#     @pipeline_model()
#     class CustomModel:
#         def __init__(self, model_path="", tokenizer_path=""):
#             self.model_path = model_path
#             self.tokenizer_path = tokenizer_path

#         @pipeline_function
#         def predict(self, input: str, **kwargs: dict) -> str:
#             return input + " lol"

#         @pipeline_function
#         def load(self, path: str) -> None:
#             print("load")

#     with Pipeline("test") as my_pipeline:
#         in_1 = my_pipeline.add_variable(str, is_input=True)
#         my_model = CustomModel()
#         my_model.predict(in_1)
#         my_pipeline.output(in_1)

#     output = Pipeline.run("test", "hey")
#     assert output == ["hey lol"]
