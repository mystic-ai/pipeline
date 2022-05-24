from datetime import datetime

import pytest
import responses
from responses import matchers

from pipeline.schemas.data import DataGet
from pipeline.schemas.file import FileGet
from pipeline.schemas.run import RunState

python_content = """
from pipeline.objects import Pipeline, Variable, pipeline_function


# Check if the decorator correctly uses __init__ and __enter__
def test_with_decorator():
    with Pipeline("test"):
        assert Pipeline._current_pipeline is not None
"""


@pytest.fixture
def api_response(url, token, bad_token, file_get_json, data_get_json, run_get_json):
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
            responses.POST,
            url + "/v2/data",
            json=data_get_json,
            status=201,
            match=[matchers.header_matcher({"Authorization": "Bearer " + token})],
        )
        rsps.add(
            responses.POST,
            url + "/v2/runs",
            json=run_get_json,
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
def data_get_json(file_get_json):
    return {
        "name": "test",
        "id": "data_test",
        "created_at": datetime.timestamp(datetime.now()),
        "hex_file": file_get_json,
    }


@pytest.fixture
def pipeline_id():
    return "pipeline_test"


@pytest.fixture()
def pipeline_get_json(pipeline_id):
    return {
        "id": pipeline_id,
        "name": "pipeline_test",
        "variables": [],
        "functions": [],
        "models": [],
        "graph_nodes": [],
        "outputs": [],
        "description": "My pipeline",
        "tags": ["featured", "NLP"],
        "public": False,
    }


@pytest.fixture()
def run_get_json(data_get_json, pipeline_get_json, pipeline_id):
    return {
        "id": pipeline_id,
        "created_at": datetime.timestamp(datetime.now()),
        "run_state": RunState.COMPLETE.value,
        "runnable": pipeline_get_json,
        "data": data_get_json,
    }
