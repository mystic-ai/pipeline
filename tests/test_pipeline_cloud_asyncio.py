import pytest

from pipeline.api.asyncio import PipelineCloud
from pipeline.exceptions.MissingActiveToken import MissingActiveToken


@pytest.mark.usefixtures("api_response")
def test_cloud_init(url, token):
    api = PipelineCloud(url=url, token=token)
    assert api.token == token


@pytest.mark.usefixtures("api_response")
def test_cloud_init_failure(url, bad_token):
    with pytest.raises(MissingActiveToken):
        PipelineCloud(url=url, token=bad_token)
