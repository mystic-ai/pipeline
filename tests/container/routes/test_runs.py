from unittest.mock import patch

from fastapi import status

from pipeline.cloud.schemas import runs as run_schemas


class TestCreateRun:

    async def test_success(self, client):

        payload = run_schemas.ContainerRunCreate(
            run_id="run_123",
            inputs=[
                run_schemas.RunInput(type="integer", value=5),
                run_schemas.RunInput(type="integer", value=4),
            ],
        )
        response = await client.post("/v4/runs", json=payload.dict())

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
        response = await client.post("/v4/runs", json=payload.dict())

        assert response.status_code == status.HTTP_200_OK
        result = run_schemas.ContainerRunResult.parse_obj(response.json())
        assert result.error is None
        assert result.outputs == [run_schemas.RunOutput(type="integer", value=15)]

    async def test_when_pipeline_failed_to_load(self, client_failed_pipeline):
        client = client_failed_pipeline
        payload = run_schemas.ContainerRunCreate(
            run_id="run_123",
            inputs=[
                run_schemas.RunInput(type="integer", value=5),
                run_schemas.RunInput(type="integer", value=4),
            ],
        )
        response = await client.post("/v4/runs", json=payload.dict())

        assert response.status_code == status.HTTP_200_OK
        result = run_schemas.ContainerRunResult.parse_obj(response.json())
        assert result.outputs is None
        assert result.error is not None
        assert result.error.type == run_schemas.ContainerRunErrorType.startup_error
        assert result.error.message == "Pipeline failed to load"
        assert result.error.traceback is not None

    async def test_when_invalid_inputs(self, client):
        payload = run_schemas.ContainerRunCreate(
            run_id="run_123",
            # one input is missing
            inputs=[
                run_schemas.RunInput(type="integer", value=5),
            ],
        )
        response = await client.post("/v4/runs", json=payload.dict())

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        result = run_schemas.ContainerRunResult.parse_obj(response.json())
        assert result.outputs is None
        assert result.error is not None
        assert result.error.type == run_schemas.ContainerRunErrorType.input_error
        assert result.error.message == "Inputs do not match graph inputs"

    async def test_when_pipeline_raises_an_exception(self, client):
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
        response = await client.post("/v4/runs", json=payload.dict())

        assert response.status_code == status.HTTP_200_OK
        result = run_schemas.ContainerRunResult.parse_obj(response.json())
        assert result.outputs is None
        assert result.error is not None
        assert result.error.type == run_schemas.ContainerRunErrorType.pipeline_error
        assert result.error.message == "ValueError('I can only sum positive integers')"
        assert result.error.traceback is not None
        assert result.error.traceback.startswith("Traceback (most recent call last):")


class TestCreateAsyncRun:

    async def test_success(self, client):
        """In the case of an async run, the API should respond immediately, and
        then make an API call to the callback URL once the run is complete.
        """
        with patch(
            "pipeline.container.services.run._send_async_result"
        ) as mock_send_async_result:
            callback_url = "https://example.com/callback"
            payload = run_schemas.ContainerRunCreate(
                run_id="run_123",
                inputs=[
                    run_schemas.RunInput(type="integer", value=5),
                    run_schemas.RunInput(type="integer", value=4),
                ],
                async_run=True,
                callback_url=callback_url,
            )
            response = await client.post("/v4/runs", json=payload.dict())

            assert response.status_code == status.HTTP_202_ACCEPTED
            result = run_schemas.ContainerRunResult.parse_obj(response.json())
            assert result.error is None
            assert result.outputs is None

            expected_response_schema = run_schemas.ContainerRunResult(
                inputs=None,
                error=None,
                outputs=[run_schemas.RunOutput(type="integer", value=9)],
            )
            mock_send_async_result.assert_called_once_with(
                callback_url=callback_url, response_schema=expected_response_schema
            )

        # make a synchronous run afterwards to ensure execution handler is still
        # working as expected
        payload = run_schemas.ContainerRunCreate(
            run_id="run_124",
            inputs=[
                run_schemas.RunInput(type="integer", value=5),
                run_schemas.RunInput(type="integer", value=10),
            ],
        )
        response = await client.post("/v4/runs", json=payload.dict())

        assert response.status_code == status.HTTP_200_OK
        result = run_schemas.ContainerRunResult.parse_obj(response.json())
        assert result.error is None
        assert result.outputs == [run_schemas.RunOutput(type="integer", value=15)]

    async def test_when_pipeline_failed_to_load(self, client_failed_pipeline):
        """Should return an error immediately in this case"""

        client = client_failed_pipeline
        with patch(
            "pipeline.container.services.run._send_async_result"
        ) as mock_send_async_result:

            payload = run_schemas.ContainerRunCreate(
                run_id="run_123",
                inputs=[
                    run_schemas.RunInput(type="integer", value=5),
                    run_schemas.RunInput(type="integer", value=4),
                ],
                async_run=True,
                callback_url="https://example.com/callback",
            )
            response = await client.post("/v4/runs", json=payload.dict())

            assert response.status_code == status.HTTP_200_OK
            result = run_schemas.ContainerRunResult.parse_obj(response.json())
            assert result.outputs is None
            assert result.error is not None
            assert result.error.type == run_schemas.ContainerRunErrorType.startup_error
            assert result.error.message == "Pipeline failed to load"
            assert result.error.traceback is not None

            mock_send_async_result.assert_not_called()

    async def test_when_invalid_inputs(self, client):
        """In the case of invalid inputs, the API will respond immediately, and
        then make an API call to the callback URL with an error.

        Perhaps it would be better to return the error immediately, but the
        parsing of inputs is currently handled at run execution time, which
        happens asynchronously.
        """
        with patch(
            "pipeline.container.services.run._send_async_result"
        ) as mock_send_async_result:
            callback_url = "https://example.com/callback"
            payload = run_schemas.ContainerRunCreate(
                run_id="run_123",
                # one input is missing
                inputs=[
                    run_schemas.RunInput(type="integer", value=5),
                ],
                async_run=True,
                callback_url=callback_url,
            )
            response = await client.post("/v4/runs", json=payload.dict())

            assert response.status_code == status.HTTP_202_ACCEPTED
            result = run_schemas.ContainerRunResult.parse_obj(response.json())
            assert result.error is None
            assert result.outputs is None

            expected_response_schema = run_schemas.ContainerRunResult(
                inputs=None,
                error=run_schemas.ContainerRunError(
                    type=run_schemas.ContainerRunErrorType.input_error,
                    message="Inputs do not match graph inputs",
                ),
                outputs=None,
            )
            mock_send_async_result.assert_called_once_with(
                callback_url=callback_url, response_schema=expected_response_schema
            )

    async def test_when_pipeline_raises_an_exception(self, client):
        """We've set up the fixture pipeline to only accept positive integers,
        so providing negative ones should result in a RunnableError.

        (Note: in reality we could add options to our inputs to handle this and
        return an input_error)
        """
        with patch(
            "pipeline.container.services.run._send_async_result"
        ) as mock_send_async_result:
            callback_url = "https://example.com/callback"
            payload = run_schemas.ContainerRunCreate(
                run_id="run_123",
                inputs=[
                    run_schemas.RunInput(type="integer", value=-5),
                    run_schemas.RunInput(type="integer", value=5),
                ],
                async_run=True,
                callback_url=callback_url,
            )
            response = await client.post("/v4/runs", json=payload.dict())

            assert response.status_code == status.HTTP_202_ACCEPTED
            result = run_schemas.ContainerRunResult.parse_obj(response.json())
            assert result.error is None
            assert result.outputs is None

            mock_calls = mock_send_async_result.call_args_list
            assert len(mock_calls) == 1
            assert mock_calls[0].kwargs["callback_url"] == callback_url
            actual_error = mock_calls[0].kwargs["response_schema"].error
            assert actual_error.type == run_schemas.ContainerRunErrorType.pipeline_error
            assert (
                actual_error.message == "ValueError('I can only sum positive integers')"
            )
            assert actual_error.traceback.startswith(
                "Traceback (most recent call last):"
            )
