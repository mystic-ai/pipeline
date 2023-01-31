# flake8: noqa
import os

os.environ["PIPELINE_CACHE"] = "./.tmp_cache/"

import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

import cloudpickle
import dill
import pytest
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Response

from pipeline.objects import (
    Graph,
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
from pipeline.schemas.pagination import Paginated
from pipeline.schemas.pipeline import (
    PipelineTagCreate,
    PipelineTagGet,
    PipelineTagPatch,
)
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


@pytest.fixture(scope="session")
def httpserver_listen_address() -> Tuple[str, int]:
    # Define the listen address for pytest-httpserver at the session scope so
    # we can use it in defining environment variables before app creation
    # https://pytest-httpserver.readthedocs.io/en/latest/howto.html#fixture
    return ("127.0.0.1", 8080)


@pytest.fixture(scope="session")
def url(httpserver_listen_address: Tuple[str, int]):
    host, port = httpserver_listen_address
    return f"http://{host}:{port}"


@pytest.fixture
def top_api_server_bad_token(httpserver: HTTPServer, bad_token: str):
    httpserver.expect_request(
        "/v2/users/me",
        method="GET",
        headers={"Authorization": "Bearer " + bad_token},
    ).respond_with_json({"auth": False}, status=401)


@pytest.fixture
def top_api_server(
    httpserver: HTTPServer,
    token: str,
    file_get: FileGet,
    tag_get: PipelineTagGet,
    tag_get_2: PipelineTagGet,
    tag_get_3: PipelineTagGet,
    tag_get_patched: PipelineTagGet,
    tag_patch: PipelineTagPatch,
    tag_create: PipelineTagCreate,
    function_get_json: dict,
    model_get_json: dict,
    result_file_get_json: dict,
    data_get_json: dict,
    pipeline_file_direct_upload_init_get_json: dict,
    pipeline_file_direct_upload_part_get_json: dict,
    finalise_direct_pipeline_file_upload_get_json: dict,
    run_get: RunGet,
    run_executing_get: RunGet,
):
    """Return an HTTP server which acts like the Top service."""

    function_get_id = function_get_json["id"]
    model_get_id = model_get_json["id"]
    result_file_get_id = result_file_get_json["id"]
    data_get_id = data_get_json["id"]

    # clear old assertions: tests where httpserver set to fail on purpose can cause
    # later tests using httpserver to fail if the error is not flushed
    httpserver.clear_assertions()

    httpserver.expect_request(
        "/v2/users/me",
        method="GET",
        headers={"Authorization": "Bearer " + token},
    ).respond_with_json({"auth": True})

    httpserver.expect_request(
        "/v2/files/",
        method="POST",
        headers={"Authorization": "Bearer " + token},
    ).respond_with_json(file_get.dict())

    httpserver.expect_request(
        f"/v2/functions/{function_get_id}",
        method="GET",
        query_string="return_data=true",
        headers={"Authorization": "Bearer " + token},
    ).respond_with_json(function_get_json)

    httpserver.expect_request(
        f"/v2/models/{model_get_id}",
        method="GET",
        query_string="return_data=true",
        headers={"Authorization": "Bearer " + token},
    ).respond_with_json(model_get_json)

    httpserver.expect_request(
        f"/v2/files/{result_file_get_id}",
        method="GET",
        query_string="return_data=true",
        headers={"Authorization": "Bearer " + token},
    ).respond_with_json(result_file_get_json)

    httpserver.expect_request(
        f"/v2/data/{data_get_id}",
        method="GET",
        query_string="return_data=true",
        headers={"Authorization": "Bearer " + token},
    ).respond_with_json(data_get_json)

    httpserver.expect_request(
        "/v2/pipeline-files/initiate-multipart-upload",
        method="POST",
        headers={"Authorization": "Bearer " + token},
    ).respond_with_json(pipeline_file_direct_upload_init_get_json)

    httpserver.expect_request(
        "/v2/pipeline-files/presigned-url",
        method="POST",
        headers={"Authorization": "Bearer " + token},
        data=json.dumps(
            {
                "pipeline_file_id": "pipeline_file_id",
                "part_num": 1,
            }
        ),
    ).respond_with_json(pipeline_file_direct_upload_part_get_json)

    httpserver.expect_request(
        "/v2/pipeline-files/finalise-multipart-upload",
        method="POST",
        headers={"Authorization": "Bearer " + token},
        data=json.dumps(
            {
                "pipeline_file_id": "pipeline_file_id",
                "multipart_metadata": [{"ETag": "dummy_etag", "PartNumber": 1}],
            }
        ),
    ).respond_with_json(finalise_direct_pipeline_file_upload_get_json)

    httpserver.expect_request(
        "/error/500",
        method="POST",
        headers={"Authorization": "Bearer " + token},
    ).respond_with_data(status=500)

    httpserver.expect_request(
        "/v2/runs",
        method="GET",
        headers={"Authorization": "Bearer " + token},
        query_string="limit=20&skip=0&order_by=created_at%3Adesc",
    ).respond_with_json(
        json.loads(
            Paginated[RunGet](
                skip=0, limit=20, total=2, data=[run_get, run_executing_get]
            ).json()
        )
    )

    httpserver.expect_request(
        f"/v2/runs/{run_get.id}",
        method="GET",
    ).respond_with_json(json.loads(run_get.json()))

    ##########
    # /v2/pipeline-tags

    httpserver.expect_request(
        "/v2/pipeline-tags",
        method="POST",
        headers=dict(Authorization=f"Bearer {token}"),
        data=tag_create.json(),
    ).respond_with_json(tag_get.dict())

    httpserver.expect_request(
        f"/v2/pipeline-tags/by-name/{tag_get.name}",
        method="GET",
        headers=dict(Authorization=f"Bearer {token}"),
    ).respond_with_json(tag_get.dict())

    httpserver.expect_request(
        f"/v2/pipeline-tags/by-name/{tag_get_2.name}",
        method="GET",
        headers=dict(Authorization=f"Bearer {token}"),
    ).respond_with_json(tag_get_2.dict())

    httpserver.expect_request(
        f"/v2/pipeline-tags/by-name/{tag_get_3.name}",
        method="GET",
        headers=dict(Authorization=f"Bearer {token}"),
    ).respond_with_json(tag_get_3.dict())

    httpserver.expect_request(
        f"/v2/pipeline-tags/by-name/missing:tag",
        method="GET",
        headers=dict(Authorization=f"Bearer {token}"),
    ).respond_with_response(Response(status=404))

    httpserver.expect_request(
        f"/v2/pipeline-tags/{tag_get.id}",
        method="PATCH",
        headers=dict(Authorization=f"Bearer {token}"),
        data=tag_patch.json(),
    ).respond_with_json(tag_get_patched.dict())

    httpserver.expect_request(
        "/v2/pipeline-tags",
        method="GET",
        headers=dict(Authorization=f"Bearer {token}"),
        query_string="skip=1&limit=5&order_by=created_at%3Adesc&pipeline_id=pipeline_id",
    ).respond_with_json(
        json.loads(
            Paginated[PipelineTagGet](
                skip=1, limit=5, total=3, data=[tag_get_2, tag_get_3]
            ).json()
        )
    )
    httpserver.expect_request(
        f"/v2/pipeline-tags/{tag_get.id}",
        method="DELETE",
        headers=dict(Authorization=f"Bearer {token}"),
    ).respond_with_response(Response(status=204))

    httpserver.expect_request(
        f"/v2/pipeline-tags/missing:tag",
        method="DELETE",
        headers=dict(Authorization=f"Bearer {token}"),
    ).respond_with_response(Response(status=404))

    ##########
    # /v2/functions

    httpserver.expect_request(
        "/v2/functions",
        method="POST",
        headers=dict(Authorization=f"Bearer {token}"),
    ).respond_with_json(function_get_json)

    return httpserver


@pytest.fixture
def data_store_httpserver() -> HTTPServer:
    server = HTTPServer(host="127.0.0.1", port=8081)
    server.start()
    server.expect_request(
        "",
        method="PUT",
    ).respond_with_data(headers={"Etag": "dummy_etag"})
    yield server
    server.clear()
    if server.is_running():
        server.stop()


@pytest.fixture()
def token() -> str:
    return "token"


@pytest.fixture()
def bad_token() -> str:
    return "bad_token"


@pytest.fixture()
def tmp_file() -> str:
    return "tests/test_model.py"


@pytest.fixture()
def serialized_function() -> str:
    def test() -> str:
        return "I'm a test!"

    return python_object_to_hex(test)


@pytest.fixture()
def file_get(serialized_function: str) -> FileGet:
    return FileGet(
        name="test",
        id="function_file_test",
        path="test/path/to/file",
        data=serialized_function,
        file_size=8,
    )


@pytest.fixture()
def file_get_json(file_get: FileGet) -> dict:
    return {
        "name": file_get.name,
        "id": file_get.id,
        "path": file_get.path,
        "data": file_get.data,
        "file_size": file_get.file_size,
    }


@pytest.fixture()
def result_file_get() -> FileGet:
    return FileGet(
        name="test_result_file",
        id="result_file_test",
        path="test/path/to/result_file",
        data=python_object_to_hex(dict(test="hello")),
        file_size=8,
    )


@pytest.fixture()
def result_file_get_json(result_file_get: FileGet) -> dict:
    return result_file_get.dict()


@pytest.fixture()
def project_get() -> ProjectGet:
    return ProjectGet(
        name="test_name",
        id="test_project_id",
        avatar_colour="#AA2216",
        avatar_image_url=None,
    )


@pytest.fixture()
def project_get_json(project_get: ProjectGet) -> dict:
    return {
        "avatar_colour": project_get.avatar_colour,
        "avatar_image_url": project_get.avatar_image_url,
        "name": project_get.name,
        "id": project_get.id,
    }


@pytest.fixture()
def function_get(file_get: FileGet, project_get: ProjectGet) -> FunctionGet:
    return FunctionGet(
        name="test_name",
        id="test_function_id",
        type=RunnableType.function,
        project=project_get,
        hex_file=file_get,
        source_sample="test_source",
    )


@pytest.fixture()
def function_get_json(
    function_get: FunctionGet, file_get_json: dict, project_get_json: dict
) -> dict:
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
def data_get(file_get: FileGet) -> DataGet:
    return DataGet(id="data_test", hex_file=file_get, created_at=datetime.now())


@pytest.fixture()
def run_get(
    function_get: FunctionGet, data_get: DataGet, result_file_get: FileGet
) -> RunGet:
    datetime.now()
    return RunGet(
        id="run_test",
        created_at=datetime(2000, 1, 1, 0, 0, 0, 0),
        run_state=RunState.COMPLETE,
        runnable=function_get,
        data=data_get,
        result=result_file_get,
    )


@pytest.fixture()
def run_executing_get(function_get: FunctionGet, data_get: DataGet) -> RunGet:
    return RunGet(
        id="run_test_2",
        created_at=datetime(2000, 1, 1, 0, 0, 0, 0),
        run_state=RunState.ALLOCATING_CLUSTER,
        runnable=function_get,
        data=data_get,
        result=None,
    )


@pytest.fixture()
def data_get_json(data_get: DataGet, file_get_json: FileGet) -> dict:
    return {
        "id": data_get.id,
        "hex_file": file_get_json,
        "created_at": str(data_get.created_at),
    }


@pytest.fixture()
def pipeline_file_direct_upload_init_get_json() -> dict:
    return PipelineFileDirectUploadInitGet(pipeline_file_id="pipeline_file_id").dict()


@pytest.fixture()
def presigned_url() -> str:
    return "http://127.0.0.1:8081"


@pytest.fixture()
def pipeline_file_direct_upload_part_get_json(presigned_url) -> dict:
    return PipelineFileDirectUploadPartGet(upload_url=presigned_url).dict()


@pytest.fixture()
def finalise_direct_pipeline_file_upload_get_json() -> dict:
    return PipelineFileGet(
        id="pipeline_file_id",
        name="pipeline_file_id",
        file=FileGet(
            name="pipeline_file_id",
            id="dummy_file_id",
            path="pipeline_file_id",
            data="dummy_data",
            file_size=10,
        ),
    ).dict()


@pytest.fixture()
def pipeline_graph() -> Graph:
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
def pickled_graph(pipeline_graph: Graph) -> dict:
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
def serialized_model(pipeline_graph: Graph) -> str:
    return python_object_to_hex(pipeline_graph.models[0].model)


@pytest.fixture()
def model_file_get(serialized_model: str) -> FileGet:
    return FileGet(
        name="test",
        id="model_file_test",
        path="test/path/to/file",
        data=serialized_model,
        file_size=8,
    )


@pytest.fixture()
def model_file_get_json(model_file_get: FileGet) -> dict:
    return {
        "name": model_file_get.name,
        "id": model_file_get.id,
        "path": model_file_get.path,
        "data": model_file_get.data,
        "file_size": model_file_get.file_size,
    }


@pytest.fixture()
def model_get(model_file_get: FileGet) -> ModelGet:
    return ModelGet(
        name="test_name",
        id="test_model_id",
        hex_file=model_file_get,
        source_sample="test_source",
    )


@pytest.fixture()
def model_get_json(model_get: ModelGet, model_file_get_json: dict) -> dict:
    return {
        "name": model_get.name,
        "id": model_get.id,
        "hex_file": model_file_get_json,
        "source_sample": model_get.source_sample,
    }


@pytest.fixture()
def pipeline_graph_with_compute_requirements() -> Graph:
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
def file(tmp_path: Path) -> Path:
    path = tmp_path / "hello.txt"
    path.write_text("hello")
    return path


@pytest.fixture()
def pipeline_file(file: Path) -> PipelineFile:
    return PipelineFile(path=str(file), name="hello")


@pytest.fixture()
def tag_get(project_get: ProjectGet) -> PipelineTagGet:
    return PipelineTagGet(
        id="pipeline_tag",
        name="test:pipeline_id",
        project_id=project_get.id,
        pipeline_id="pipeline_id",
    )


@pytest.fixture()
def tag_get_2(project_get: ProjectGet) -> PipelineTagGet:
    return PipelineTagGet(
        id="pipeline_tag_2",
        name="test:tag2",
        project_id=project_get.id,
        pipeline_id="pipeline_id",
    )


@pytest.fixture()
def tag_get_3(project_get: ProjectGet) -> PipelineTagGet:
    return PipelineTagGet(
        id="pipeline_tag_3",
        name="test:tag3",
        project_id=project_get.id,
        pipeline_id="pipeline_id_2",
    )


@pytest.fixture()
def tag_get_patched(project_get: ProjectGet) -> PipelineTagGet:
    return PipelineTagGet(
        id="pipeline_tag",
        name="test:pipeline_id",
        project_id=project_get.id,
        pipeline_id="pipeline_id_2",
    )


@pytest.fixture()
def tag_create() -> PipelineTagCreate:
    return PipelineTagCreate(
        name="test:pipeline_id",
        pipeline_id="pipeline_id",
    )


@pytest.fixture()
def tag_patch() -> PipelineTagPatch:
    return PipelineTagPatch(
        pipeline_id="pipeline_id_2",
    )


@pytest.fixture()
def tags_list(
    tag_get_2: PipelineTagGet,
    tag_get_3: PipelineTagGet,
) -> Paginated[PipelineTagGet]:
    return Paginated[PipelineTagGet](
        skip=1, limit=5, total=3, data=[tag_get_2, tag_get_3]
    )
