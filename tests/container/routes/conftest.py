import pytest
from fastapi.testclient import TestClient

from pipeline.container.startup import create_app


@pytest.fixture
async def app():
    app = create_app()
    return app


@pytest.fixture
async def client(app):
    with TestClient(app) as client:
        yield client
