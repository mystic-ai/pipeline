import http

import pytest

from pipeline import PipelineCloud


@pytest.mark.usefixtures("api_response")
def test_cloud_init(url, token):
    api = PipelineCloud(url, token)
    assert api.token == token


@pytest.mark.usefixtures("api_response")
def test_cloud_init_failure(url, bad_token):
    with pytest.raises(Exception) as e:
        PipelineCloud(url, bad_token)
        assert e.status == http.HTTPStatus.UNAUTHORIZED


@pytest.mark.usefixtures("api_response")
def test_cloud_upload_file(url, token, file_get, tmp_file):
    api = PipelineCloud(url, token)
    f = api.upload_file(tmp_file, "remote_path")
    assert f == file_get
