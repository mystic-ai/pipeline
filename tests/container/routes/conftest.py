from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

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


@pytest.fixture
async def mock_manager():
    with patch("pipeline.container.startup.Manager", DummyManager) as mock:
        yield mock


@pytest.fixture
async def app(mock_manager):
    app = create_app()
    return app


@pytest.fixture
async def client(app):
    with TestClient(app) as client:
        yield client
