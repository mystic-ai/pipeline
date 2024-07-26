from fastapi import status

from pipeline.cloud.schemas import runs as run_schemas


async def test_create_run(client):
    payload = run_schemas.ContainerRunCreate(
        run_id="run_123",
        inputs=[run_schemas.RunInput(type="string", value="foo")],
    )
    response = client.post("/v4/runs", json=payload.dict())

    assert response.status_code == status.HTTP_200_OK
