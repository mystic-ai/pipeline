# flake8: noqa
from datetime import datetime

import cloudpickle
import dill
import pytest
import responses
from responses import matchers

from pipeline.objects import (
    Pipeline,
    PipelineFile,
    Variable,
    pipeline_function,
    pipeline_model,
)
from pipeline.schemas.data import DataGet
from pipeline.schemas.file import FileGet
from pipeline.schemas.function import FunctionGet
from pipeline.schemas.model import ModelGet
from pipeline.schemas.pipeline_file import (
    PipelineFileDirectUploadInitGet,
    PipelineFileDirectUploadPartGet,
    PipelineFileGet,
)
from pipeline.schemas.project import ProjectGet
from pipeline.schemas.run import RunGet, RunState
from pipeline.schemas.runnable import RunnableType
from pipeline.util import python_object_to_hex

python_content = """
from pipeline.objects import Pipeline, Variable, pipeline_function


# Check if the decorator correctly uses __init__ and __enter__
def test_with_decorator():
    with Pipeline("test"):
        assert Pipeline._current_pipeline is not None
"""


@pytest.fixture
def api_response(
    url,
    token,
    bad_token,
    file_get_json,
    function_get_json,
    result_file_get_json,
    model_get_json,
    data_get_json,
    pipeline_file_direct_upload_init_get_json,
    pipeline_file_direct_upload_part_get_json,
    presigned_url,
    finalise_direct_pipeline_file_upload_get_json,
):
    function_get_id = function_get_json["id"]
    model_get_id = model_get_json["id"]
    data_get_id = data_get_json["id"]
    result_file_get_id = result_file_get_json["id"]
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
        rsps.add(
            responses.GET,
            url + f"/v2/functions/{function_get_id}",
            json=function_get_json,
            status=200,
            match=[matchers.header_matcher({"Authorization": "Bearer " + token})],
        )
        rsps.add(
            responses.GET,
            url + f"/v2/models/{model_get_id}",
            json=model_get_json,
            status=200,
            match=[matchers.header_matcher({"Authorization": "Bearer " + token})],
        )
        rsps.add(
            responses.GET,
            url + f"/v2/data/{data_get_id}",
            json=data_get_json,
            status=200,
            match=[matchers.header_matcher({"Authorization": "Bearer " + token})],
        )
        rsps.add(
            responses.GET,
            url + f"/v2/files/{result_file_get_id}",
            json=result_file_get_json,
            status=200,
            match=[matchers.header_matcher({"Authorization": "Bearer " + token})],
        )
        rsps.add(
            responses.POST,
            url + "/v2/pipeline-files/initiate-multipart-upload",
            json=pipeline_file_direct_upload_init_get_json,
            status=200,
            match=[matchers.header_matcher({"Authorization": "Bearer " + token})],
        )
        rsps.add(
            responses.POST,
            url + "/v2/pipeline-files/presigned-url",
            json=pipeline_file_direct_upload_part_get_json,
            status=200,
            match=[
                matchers.header_matcher({"Authorization": "Bearer " + token}),
                matchers.json_params_matcher(
                    {
                        "pipeline_file_id": "pipeline_file_id",
                        "part_num": 1,
                    }
                ),
            ],
        )
        # upload file directly using presigned url
        rsps.add(
            responses.PUT, presigned_url, status=200, headers={"Etag": "dummy_etag"}
        )
        rsps.add(
            responses.POST,
            url + "/v2/pipeline-files/finalise-multipart-upload",
            json=finalise_direct_pipeline_file_upload_get_json,
            status=200,
            match=[
                matchers.header_matcher({"Authorization": "Bearer " + token}),
                matchers.json_params_matcher(
                    {
                        "pipeline_file_id": "pipeline_file_id",
                        "multipart_metadata": [{"ETag": "dummy_etag", "PartNumber": 1}],
                    }
                ),
            ],
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
def serialized_function():
    def test() -> str:
        return "I'm a test!"

    return python_object_to_hex(test)


@pytest.fixture()
def file_get(serialized_function):
    return FileGet(
        name="test",
        id="function_file_test",
        path="test/path/to/file",
        data=serialized_function,
        file_size=8,
    )


@pytest.fixture()
def file_get_json(file_get):
    return {
        "name": file_get.name,
        "id": file_get.id,
        "path": file_get.path,
        "data": file_get.data,
        "file_size": file_get.file_size,
    }


@pytest.fixture()
def result_file_get():
    return FileGet(
        name="test_result_file",
        id="result_file_test",
        path="test/path/to/result_file",
        data=python_object_to_hex(dict(test="hello")),
        file_size=8,
    )


@pytest.fixture()
def result_file_get_json(result_file_get):
    return {
        "name": result_file_get.name,
        "id": result_file_get.id,
        "path": result_file_get.path,
        "data": result_file_get.data,
        "file_size": result_file_get.file_size,
    }


@pytest.fixture()
def project_get():
    return ProjectGet(
        name="test_name",
        id="test_project_id",
        avatar_colour="#AA2216",
        avatar_image_url=None,
    )


@pytest.fixture()
def project_get_json(project_get):
    return {
        "avatar_colour": project_get.avatar_colour,
        "avatar_image_url": project_get.avatar_image_url,
        "name": project_get.name,
        "id": project_get.id,
    }


@pytest.fixture()
def function_get(file_get, project_get):
    return FunctionGet(
        name="test_name",
        id="test_function_id",
        type=RunnableType.function,
        project=project_get,
        hex_file=file_get,
        source_sample="test_source",
    )


@pytest.fixture()
def function_get_json(function_get, file_get_json, project_get_json):
    return {
        "id": function_get.id,
        "type": function_get.type.value,
        "name": function_get.name,
        "project": project_get_json,
        "hex_file": file_get_json,
        "source_sample": function_get.source_sample,
        "inputs": [],
        "output": [],
    }


@pytest.fixture()
def data_get(file_get):
    return DataGet(id="data_test", hex_file=file_get, created_at=datetime.now())


@pytest.fixture()
def run_get(function_get, data_get, result_file_get):
    return RunGet(
        id="run_test",
        created_at=datetime.now(),
        run_state=RunState.COMPLETE,
        runnable=function_get,
        data=data_get,
        result=result_file_get,
    )


@pytest.fixture()
def data_get_json(data_get, file_get_json):
    return {
        "id": data_get.id,
        "hex_file": file_get_json,
        "created_at": str(data_get.created_at),
    }


@pytest.fixture()
def pipeline_file_direct_upload_init_get_json():
    return PipelineFileDirectUploadInitGet(pipeline_file_id="pipeline_file_id").dict()


@pytest.fixture()
def presigned_url():
    return "https://upload-file-here.com"


@pytest.fixture()
def pipeline_file_direct_upload_part_get_json(presigned_url):
    return PipelineFileDirectUploadPartGet(upload_url=presigned_url).dict()


@pytest.fixture()
def finalise_direct_pipeline_file_upload_get_json():
    return PipelineFileGet(
        id="pipeline_file_id",
        name="pipeline_file_id",
        hex_file=FileGet(
            name="pipeline_file_id",
            id="dummy_file_id",
            path="pipeline_file_id",
            data="dummy_data",
            file_size=10,
        ),
    ).dict()


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
                    "data": "80049527000000000000008c0a64696c6c2e5f64696c6c948c0a5f6c6f61645f747970659493948c0373747294859452942e",
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
                    "data": "80049527000000000000008c0a64696c6c2e5f64696c6c948c0a5f6c6f61645f747970659493948c0373747294859452942e",
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
                    "data": dill.dumps(pipeline_graph.functions[0]).hex(),
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


@pytest.fixture()
def serialized_model(pipeline_graph):
    return python_object_to_hex(pipeline_graph.models[0].model)


@pytest.fixture()
def model_file_get(serialized_model):
    return FileGet(
        name="test",
        id="model_file_test",
        path="test/path/to/file",
        data=serialized_model,
        file_size=8,
    )


@pytest.fixture()
def model_file_get_json(model_file_get):
    return {
        "name": model_file_get.name,
        "id": model_file_get.id,
        "path": model_file_get.path,
        "data": model_file_get.data,
        "file_size": model_file_get.file_size,
    }


@pytest.fixture()
def model_get(model_file_get):
    return ModelGet(
        name="test_name",
        id="test_model_id",
        hex_file=model_file_get,
        source_sample="test_source",
    )


@pytest.fixture()
def model_get_json(model_get, model_file_get_json):
    return {
        "name": model_get.name,
        "id": model_get.id,
        "hex_file": model_file_get_json,
        "source_sample": model_get.source_sample,
    }


@pytest.fixture()
def pipeline_graph_with_compute_requirements():
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

    with Pipeline("test", compute_type="gpu", min_gpu_vram_mb=4000) as my_pipeline:
        in_1 = Variable(str, is_input=True)
        my_pipeline.add_variable(in_1)

        my_model = CustomModel()
        str_1 = my_model.predict(in_1)

        my_pipeline.output(str_1)
    return Pipeline.get_pipeline("test")


@pytest.fixture()
def file(tmp_path):
    path = tmp_path / "hello.txt"
    path.write_text("hello")
    return path


@pytest.fixture()
def pipeline_file(file):
    return PipelineFile(path=str(file), name="hello")
