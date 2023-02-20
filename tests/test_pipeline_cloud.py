import httpx
import pytest

from pipeline import Pipeline, PipelineCloud, PipelineFile
from pipeline.exceptions.InvalidSchema import InvalidSchema
from pipeline.exceptions.MissingActiveToken import MissingActiveToken
from pipeline.schemas.file import FileGet
from pipeline.util import hex_to_python_object


@pytest.mark.usefixtures("top_api_server")
def test_cloud_init(url, token):
    api = PipelineCloud(url=url, token=token)
    assert api.token == token


def test_cloud_init_failure(url, top_api_server_bad_token, bad_token):
    with pytest.raises(MissingActiveToken):
        PipelineCloud(url=url, token=bad_token)


@pytest.mark.usefixtures("top_api_server")
def test_cloud_upload_function_fail(url, token):
    api = PipelineCloud(url=url, token=token)
    with pytest.raises(InvalidSchema):
        api.upload_function("")


@pytest.mark.usefixtures("top_api_server")
def test_cloud_download_function(url, token, function_get, file_get):
    api = PipelineCloud(url=url, token=token)
    f = api.download_function(function_get.id)
    assert f.function() == hex_to_python_object(file_get.data)()


@pytest.mark.usefixtures("top_api_server")
def test_cloud_download_model(url, token, model_get, model_file_get):
    api = PipelineCloud(url=url, token=token)
    m = api.download_model(model_get.id)
    model_class = type(hex_to_python_object(model_file_get.data))
    assert isinstance(m.model, model_class)


@pytest.mark.usefixtures("top_api_server")
def test_cloud_download_result_via_run_get_result_id(
    url, token, run_get, result_file_get
):
    api = PipelineCloud(url=url, token=token)
    result = api.download_result(run_get.result.id)
    assert result == hex_to_python_object(result_file_get.data)


@pytest.mark.usefixtures("top_api_server")
def test_cloud_download_result_via_run_get(url, token, run_get, result_file_get):
    api = PipelineCloud(url=url, token=token)
    result = api.download_result(run_get)
    assert result == hex_to_python_object(result_file_get.data)


@pytest.mark.usefixtures("top_api_server")
def test_cloud_download_data(url, token, data_get, file_get):
    api = PipelineCloud(url=url, token=token)
    d = api.download_data(data_get.id)
    assert d() == hex_to_python_object(file_get.data)()


@pytest.mark.usefixtures("top_api_server")
@pytest.mark.usefixtures("data_store_httpserver")
def test_cloud_upload_pipeline_file(
    url,
    token,
    pipeline_file,
    file,
    finalise_direct_pipeline_file_upload_get_json,
):
    api = PipelineCloud(url=url, token=token)
    pipeline_file_var_get = api.upload_pipeline_file(pipeline_file)
    assert pipeline_file_var_get.path == str(file)
    assert (
        pipeline_file_var_get.file.dict()
        == finalise_direct_pipeline_file_upload_get_json["file"]
    )


@pytest.mark.usefixtures("top_api_server")
def test_cloud_get_raise_for_status_when_non_json_error(url, token):
    api = PipelineCloud(url=url, token=token)
    with pytest.raises(httpx.HTTPError, match="500 INTERNAL SERVER ERROR"):
        api._post("/error/500", json_data={})


@pytest.mark.usefixtures("top_api_server")
@pytest.mark.usefixtures("data_store_httpserver")
def test_remote_file_downloaded(
    url,
    token,
    result_file_get: FileGet,
):
    pcloud = PipelineCloud(url=url, token=token)
    with Pipeline("test") as builder:
        test_file = PipelineFile(remote_id=result_file_get.id)
        builder.add_variables(test_file)

    test_pipeline = Pipeline.get_pipeline("test")
    pcloud.download_remotes(test_pipeline)
    pipeline_file: PipelineFile = test_pipeline.variables[0]
    assert pipeline_file.path is not None
