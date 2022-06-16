import pytest

from pipeline import PipelineCloud
from pipeline.exceptions.InvalidSchema import InvalidSchema
from pipeline.exceptions.MissingActiveToken import MissingActiveToken
from pipeline.util import hex_to_python_object


@pytest.mark.usefixtures("api_response")
def test_cloud_init(url, token):
    api = PipelineCloud(url, token)
    assert api.token == token


@pytest.mark.usefixtures("api_response")
def test_cloud_init_failure(url, bad_token):
    with pytest.raises(MissingActiveToken):
        PipelineCloud(url, bad_token)


@pytest.mark.usefixtures("api_response")
def test_cloud_upload_file(url, token, file_get, tmp_file):
    api = PipelineCloud(url, token)
    f = api.upload_file(tmp_file, "remote_path")
    # recover data field just to simplify assertion
    f.data = hex_to_python_object(f.data)
    assert f == file_get


@pytest.mark.usefixtures("api_response")
def test_cloud_upload_function_fail(url, token):
    api = PipelineCloud(url, token)
    with pytest.raises(InvalidSchema):
        api.upload_function("")


@pytest.mark.usefixtures("api_response")
def test_cloud_download_function(url, token, function_get, file_get):
    api = PipelineCloud(url, token)
    f = api.download_function(function_get.id)
    assert f == file_get.data
