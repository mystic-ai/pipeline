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


@pytest.mark.usefixtures("api_response")
def test_cloud_repeat_run(url, token, data_get, pipeline_id):
    api = PipelineCloud(url, token)
    runs = api.run_pipeline(pipeline_id, data_get.id, repeat=3)
    assert len(runs) == 3
    assert runs[0]["id"] == pipeline_id
