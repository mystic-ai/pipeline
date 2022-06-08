# flake8: noqa
from datetime import datetime

import cloudpickle
import pytest
import responses
from responses import matchers

from pipeline.objects import Pipeline, Variable, pipeline_function, pipeline_model
from pipeline.schemas.data import DataGet
from pipeline.schemas.file import FileGet

python_content = """
from pipeline.objects import Pipeline, Variable, pipeline_function


# Check if the decorator correctly uses __init__ and __enter__
def test_with_decorator():
    with Pipeline("test"):
        assert Pipeline._current_pipeline is not None
"""


@pytest.fixture
def api_response(url, token, bad_token, file_get_json):
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        rsps.add(
            responses.GET,
            url + "/v2/users/me",
            json={"auth": True},
            status=200,
            match=[matchers.header_matcher({"Authorization": "Bearer " + token})],
        )
        rsps.add(
            responses.GET,
            url + "/v2/users/me",
            json={"auth": True},
            status=401,
            match=[matchers.header_matcher({"Authorization": "Bearer " + bad_token})],
        )
        rsps.add(
            responses.POST,
            url + "/v2/files/",
            json=file_get_json,
            status=201,
            match=[matchers.header_matcher({"Authorization": "Bearer " + token})],
        )
        yield rsps


@pytest.fixture()
def url():
    return "http://127.0.0.1:8080"


@pytest.fixture()
def token():
    return "token"


@pytest.fixture()
def bad_token():
    return "bad_token"


@pytest.fixture()
def tmp_file():
    return "tests/test_model.py"


@pytest.fixture()
def file_get_json():
    return {
        "name": "test",
        "id": "file_test",
        "path": "test/path/to/file",
        "data": "data",
        "file_size": 8,
    }


@pytest.fixture()
def file_get():
    return FileGet(
        name="test", id="file_test", path="test/path/to/file", data="data", file_size=8
    )


@pytest.fixture()
def data_get(file_get):
    return DataGet(id="data_test", hex_file=file_get, created_at=datetime.now())


@pytest.fixture()
def pipeline_graph():
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
    return Pipeline.get_pipeline("test")


@pytest.fixture()
def pickled_graph(pipeline_graph):
    return {
        "id": "pipeline_72c96d162d3347c38f83e56ce982455b",
        "type": "pipeline",
        "name": pipeline_graph.name,
        "project": {
            "avatar_colour": "#AA2216",
            "avatar_image_url": None,
            "name": "Default",
            "id": "project_3b0253647cc844619ad5b5288af40e7d",
        },
        "variables": [
            {
                "local_id": "lpacWxZNeq",
                "name": None,
                "type_file": {
                    "name": "jaahNwIiBykPFqgeXQJG",
                    "id": "file_ec28640d32b947308b34248c4bb87aeb",
                    "path": "object_dy8tcccflt315cmzgu3pivyw4w7h5l4im4wxefbexz4jxsdg5fdkqc80qpw2sxyx",
                    "data": cloudpickle.dumps(
                        pipeline_graph.variables[0].type_class
                    ).hex(),
                    "file_size": 100,
                },
                "type_file_id": None,
                "is_input": True,
                "is_output": False,
            },
            {
                "local_id": "aDhXvxVeKl",
                "name": None,
                "type_file": {
                    "name": "nAJGkyZMSlvIJmGTlkMp",
                    "id": "file_79b7384d35e54692939561a4e7e28a2e",
                    "path": "object_yrk5vc6tuowny9ysglpzdhuqm73j2wb50xjgmv9nio4n6jvh0vjrzpnipcqzncdu",
                    "data": cloudpickle.dumps(
                        pipeline_graph.variables[0].type_class
                    ).hex(),
                    "file_size": 100,
                },
                "type_file_id": None,
                "is_input": False,
                "is_output": True,
            },
        ],
        "functions": [
            {
                "id": "function_a66fae56c86e40769b4b01481b83c9b0",
                "type": "function",
                "name": "predict",
                "project": {
                    "avatar_colour": "#AA2216",
                    "avatar_image_url": None,
                    "name": "Default",
                    "id": "project_3b0253647cc844619ad5b5288af40e7d",
                },
                "hex_file": {
                    "name": "LZpywTMKOoFwqOKNMEqB",
                    "id": "file_9ec0d8d609f54e8daaa9fb4d1f2291f5",
                    "path": "object_y1u1tgfvxb2t8jojo4owted8i09nijaae9ndnsmtn0ci4pjnnc3s3c61lxo7ld3v",
                    "data": cloudpickle.dumps(pipeline_graph.functions[0]).hex(),
                    "file_size": 6618,
                },
                "source_sample": '    @pipeline_function\n    def predict(self, input_data: str, model_kwargs: dict = {}) -> str:\n        input_ids = self.tokenizer(input_data, return_tensors="pt").input_ids\n        gen_tokens = self.m',
            }
        ],
        "models": [
            {
                "id": "model_10d70951463e474a9126ec6e857249cb",
                "name": "",
                "hex_file": {
                    "name": "bfdVXaRkimcSfnDQpQGR",
                    "id": "file_6c42a41051fb42f68a3d43440cd0c2a4",
                    "path": "object_9h5z82oyjrjgx9f89q5yuenipjrypg14yrj3kx0bitviu7nqpmgx66tyqbxn0qsd",
                    "data": cloudpickle.dumps(pipeline_graph.models[0]).hex(),
                    "file_size": 9338,
                },
                "source_sample": '@pipeline_model\nclass TransformersModelForCausalLM:\n    def __init__(\n        self,\n        model_path: str = "EleutherAI/gpt-neo-125M",\n        tokenizer_path: str = "EleutherAI/gpt-neo-125M",\n    ):',
            }
        ],
        "graph_nodes": [
            {
                "local_id": "SbSVsUkIkc",
                "function": "function_a66fae56c86e40769b4b01481b83c9b0",
                "inputs": ["lpacWxZNeq"],
                "outputs": ["aDhXvxVeKl"],
            }
        ],
        "outputs": ["aDhXvxVeKl"],
    }
