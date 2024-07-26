import pytest
from fastapi import status
from fastapi.testclient import TestClient

from pipeline.cloud.schemas import runs as run_schemas


@pytest.fixture
async def client(app, monkeypatch):
    # loads a pipeline which sums 2 inputs
    monkeypatch.setenv(
        "PIPELINE_PATH", "tests.container.fixtures.adder_pipeline:my_pipeline"
    )
    with TestClient(app) as client:
        yield client


async def test_create_run(client):

    payload = run_schemas.ContainerRunCreate(
        run_id="run_123",
        inputs=[
            run_schemas.RunInput(type="integer", value=5),
            run_schemas.RunInput(type="integer", value=4),
        ],
    )
    response = client.post("/v4/runs", json=payload.dict())

    assert response.status_code == status.HTTP_200_OK
    result = run_schemas.ContainerRunResult.parse_obj(response.json())
    assert result.error is None
    assert result.outputs == [run_schemas.RunOutput(type="integer", value=9)]

    # make another run to ensure execution handler is still working as expected
    payload = run_schemas.ContainerRunCreate(
        run_id="run_124",
        inputs=[
            run_schemas.RunInput(type="integer", value=5),
            run_schemas.RunInput(type="integer", value=10),
        ],
    )
    response = client.post("/v4/runs", json=payload.dict())

    assert response.status_code == status.HTTP_200_OK
    result = run_schemas.ContainerRunResult.parse_obj(response.json())
    assert result.error is None
    assert result.outputs == [run_schemas.RunOutput(type="integer", value=15)]
