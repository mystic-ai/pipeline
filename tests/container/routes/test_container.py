from fastapi import status

from pipeline.cloud.schemas import pipelines as pipeline_schemas


async def test_is_ready(client):
    response = await client.get("/v4/container/state")
    assert response.status_code == status.HTTP_200_OK
    result = pipeline_schemas.PipelineContainerState.parse_obj(response.json())
    assert result.state == pipeline_schemas.PipelineState.loaded
    assert result.message is None
    assert result.current_run_id is None


async def test_is_ready_pipeline_load_failed(client_failed_pipeline):
    client = client_failed_pipeline
    response = await client.get("/v4/container/state")
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    result = pipeline_schemas.PipelineContainerState.parse_obj(response.json())
    assert result.state == pipeline_schemas.PipelineState.load_failed
    assert result.message is not None
    assert result.message.startswith("Traceback")
    assert result.current_run_id is None
