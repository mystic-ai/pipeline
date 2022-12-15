import pytest

from pipeline.api.asyncio import PipelineCloud
from pipeline.exceptions.MissingActiveToken import MissingActiveToken


def test_cloud_init(url, top_api_server, token):
    api = PipelineCloud(url=url, token=token)
    assert api.token == token


def test_cloud_init_failure(url, top_api_server_bad_token, bad_token):
    with pytest.raises(MissingActiveToken):
        PipelineCloud(url=url, token=bad_token)
