from fastapi import status

from pipeline.cloud.schemas import runs as run_schemas


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
    # we have mocked the run manager to sum the inputs
    assert result.outputs == [run_schemas.RunOutput(type="integer", value=9)]

    # make another run to ensure execution handler is still working as expected
    payload = run_schemas.ContainerRunCreate(
        run_id="run_123",
        inputs=[
            run_schemas.RunInput(type="integer", value=5),
            run_schemas.RunInput(type="integer", value=10),
        ],
    )
    response = client.post("/v4/runs", json=payload.dict())

    assert response.status_code == status.HTTP_200_OK
    result = run_schemas.ContainerRunResult.parse_obj(response.json())
    assert result.outputs == [run_schemas.RunOutput(type="integer", value=15)]
