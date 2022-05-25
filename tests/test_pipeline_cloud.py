import pytest

from pipeline import PipelineCloud
from pipeline.exceptions.InvalidSchema import InvalidSchema
from pipeline.exceptions.MissingActiveToken import MissingActiveToken


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
    assert f == file_get


@pytest.mark.usefixtures("api_response")
def test_cloud_upload_function_fail(url, token):
    api = PipelineCloud(url, token)
    with pytest.raises(InvalidSchema):
        api.upload_function("")
