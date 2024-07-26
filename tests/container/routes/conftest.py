from unittest.mock import patch

import pytest

# from fastapi.testclient import TestClient
from httpx import AsyncClient

from pipeline.cloud.schemas import pipelines as pipeline_schemas
from pipeline.cloud.schemas import runs as run_schemas
from pipeline.container.startup import create_app


class DummyManager:
    def __init__(self, pipeline_path):
        self.pipeline_state = pipeline_schemas.PipelineState.not_loaded

    def startup(self):
        self.pipeline_state = pipeline_schemas.PipelineState.loaded

    def run(self, run_id: str | None, input_data: list[run_schemas.RunInput] | None):
        return []


@pytest.fixture(scope="session")
async def mock_manager():
    # print(f"mock manager fixture; loop={id(asyncio.get_running_loop())}")
    with patch("pipeline.container.startup.Manager", DummyManager) as mock:
        yield mock


# not quite sure why but session scope needed to ensure event loop is shared
# across fixtures and tests
# think this may be addressed in https://github.com/pytest-dev/pytest-asyncio/pull/871
@pytest.fixture(scope="session")
async def app(mock_manager):
    # print(f"app fixture; loop={id(asyncio.get_running_loop())}")
    app = create_app()
    yield app
    app.state.execution_task.cancel()


@pytest.fixture(scope="session")
async def client(app):
    # print(f"client fixture; loop={id(asyncio.get_running_loop())}")
    # with TestClient(app) as client:
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
