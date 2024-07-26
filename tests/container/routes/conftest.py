from contextlib import asynccontextmanager

import pytest
from asgi_lifespan import LifespanManager
from httpx import AsyncClient

from pipeline.container.startup import create_app


@pytest.fixture
async def app():
    app = create_app()
    return app


@pytest.fixture
async def get_test_client(app, monkeypatch):

    @asynccontextmanager
    async def _get_test_client(pipeline_path):
        monkeypatch.setenv("PIPELINE_PATH", pipeline_path)
        async with LifespanManager(app) as manager:
            async with AsyncClient(app=manager.app, base_url="http://test") as client:
                yield client

    return _get_test_client


@pytest.fixture
async def client(get_test_client):
    async with get_test_client(
        pipeline_path="tests.container.fixtures.adder_pipeline:my_pipeline"
    ) as client:
        yield client
