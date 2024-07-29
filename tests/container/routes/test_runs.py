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


class TestCreateRun:

    def test_success(self, client):

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

    def test_when_pipeline_failed_to_load(self, app, monkeypatch):
        # use invalid path to simulate pipeline failed error
        monkeypatch.setenv("PIPELINE_PATH", "tests.container.fixtures.oops:my_pipeline")
        with TestClient(app) as client:

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
            assert result.outputs is None
            assert result.error is not None
            error = run_schemas.ContainerRunError.parse_obj(result.error)
            assert error.type == run_schemas.ContainerRunErrorType.startup_error
            assert error.message == "Pipeline failed to load"
            assert error.traceback is not None

    def test_when_invalid_inputs(self, client):
        payload = run_schemas.ContainerRunCreate(
            run_id="run_123",
            # one input is missing
            inputs=[
                run_schemas.RunInput(type="integer", value=5),
            ],
        )
        response = client.post("/v4/runs", json=payload.dict())

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        result = run_schemas.ContainerRunResult.parse_obj(response.json())
        assert result.outputs is None
        assert result.error is not None
        error = run_schemas.ContainerRunError.parse_obj(result.error)
        assert error.type == run_schemas.ContainerRunErrorType.input_error
        assert error.message == "Inputs do not match graph inputs"

    def test_when_pipeline_raises_an_exception(self, client):
        """We've set up the fixture pipeline to only accept positive integers,
        so providing negative ones should result in a RunnableError.

        (Note: in reality we could add options to our inputs to handle this and
        return an input_error)
        """
        payload = run_schemas.ContainerRunCreate(
            run_id="run_123",
            inputs=[
                run_schemas.RunInput(type="integer", value=-5),
                run_schemas.RunInput(type="integer", value=5),
            ],
        )
        response = client.post("/v4/runs", json=payload.dict())

        assert response.status_code == status.HTTP_200_OK
        result = run_schemas.ContainerRunResult.parse_obj(response.json())
        assert result.outputs is None
        assert result.error is not None
        error = run_schemas.ContainerRunError.parse_obj(result.error)
        assert error.type == run_schemas.ContainerRunErrorType.pipeline_error
        assert error.message == "ValueError('I can only sum positive integers')"
        assert error.traceback is not None
        assert error.traceback.startswith("Traceback (most recent call last):")
