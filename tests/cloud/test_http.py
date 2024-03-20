import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pipeline.cloud.http import StreamingResponseWithStatusCode


@pytest.fixture
def dummy_app():
    app = FastAPI()

    @app.get("/stream/{status_code}")
    def stream(status_code: int):
        """Dummy endpoint to return a streaming response with status code.

        This mimics beahviour when proxying a streaming response, where the
        upstream status code is unknown ahead of time (i.e. we can't just set
        the status code on the response itself as it comes from the content
        stream).
        """
        content_stream = iter(
            [
                ("Hello ", status_code),
                ("World", status_code),
            ]
        )
        return StreamingResponseWithStatusCode(
            content=content_stream,
            headers={"X-Accel-Buffering": "no"},
        )

    return app


@pytest.mark.parametrize("status_code", [200, 404, 500])
def test_streaming_response_with_status_code(status_code, dummy_app):
    """Test custom response class by setting up a dummy API"""
    client = TestClient(dummy_app)
    with client.stream("GET", f"/stream/{status_code}") as response:
        assert response.status_code == status_code
        # check headers are sent correctly
        assert response.headers["X-Accel-Buffering"] == "no"
        # check full response content
        response_data = ""
        for chunk in response.iter_text():
            response_data += chunk
        assert response_data == "Hello World"
