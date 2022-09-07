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
    assert f.function() == hex_to_python_object(file_get.data)()


@pytest.mark.usefixtures("api_response")
def test_cloud_download_model(url, token, model_get, model_file_get):
    api = PipelineCloud(url, token)
    m = api.download_model(model_get.id)
    model_class = type(hex_to_python_object(model_file_get.data))
    assert isinstance(m.model, model_class)


@pytest.mark.usefixtures("api_response")
def test_cloud_download_data(url, token, data_get, file_get):
    api = PipelineCloud(url, token)
    d = api.download_data(data_get.id)
    assert d() == hex_to_python_object(file_get.data)()


@pytest.mark.usefixtures("api_response")
def test_cloud_upload_pipeline_file(
    url, token, pipeline_file, file, finalise_direct_pipeline_file_upload_get_json
):
    api = PipelineCloud(url, token)
    pipeline_file_var_get = api.upload_pipeline_file(pipeline_file)
    assert pipeline_file_var_get.path == str(file)
    assert (
        pipeline_file_var_get.file.dict()
        == finalise_direct_pipeline_file_upload_get_json["hex_file"]
    )
