import pytest
from fastapi import status


@pytest.mark.asyncio(scope="session")
async def test_status(client):
    response = await client.get("/status")
    assert response.status_code == status.HTTP_200_OK
