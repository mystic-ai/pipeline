import json

from pipeline.cloud.schemas import runs as run_schemas
from pipeline.container.routes.v4.runs import _stream_run_outputs
from pipeline.objects.graph import Stream


class DummyRequest:
    """A dummy request object for use in this test"""

    async def is_disconnected(self):
        return False


async def test_stream_run_outputs():
    """Test that the order of outputs is as expected when we have a combination
    of static and stream outputs. Note that the stream outputs return different
    amounts of data.
    """
    stream_output_one = run_schemas.RunOutput(
        type=run_schemas.RunIOType.stream, value=Stream(iter([1, 2, 3, 4])), file=None
    )
    stream_output_two = run_schemas.RunOutput(
        type=run_schemas.RunIOType.stream,
        value=Stream(iter(["hello", "world"])),
        file=None,
    )
    static_output = run_schemas.RunOutput(
        type=run_schemas.RunIOType.string, value="static output", file=None
    )
    container_run_result = run_schemas.ContainerRunResult(
        inputs=None,
        outputs=[stream_output_one, static_output, stream_output_two],
        error=None,
    )

    results = [
        result
        async for result in _stream_run_outputs(container_run_result, DummyRequest())
    ]

    output_values = []
    for result, status_code in results:
        assert status_code == 200
        outputs = json.loads(result)["outputs"]
        values = [o["value"] for o in outputs]
        output_values.append(values)

    assert output_values == [
        [
            1,
            "static output",
            "hello",
        ],
        [
            2,
            "static output",
            "world",
        ],
        [
            3,
            "static output",
            None,
        ],
        [
            4,
            "static output",
            None,
        ],
    ]


async def test_stream_run_outputs_when_exception_raised():
    """Test streaming outputs when pipeline raises an exception.

    Error should be reported back to the user.
    """

    def error_stream():
        yield 1
        raise Exception("dummy error")

    stream_output_one = run_schemas.RunOutput(
        type=run_schemas.RunIOType.stream, value=Stream(error_stream()), file=None
    )
    stream_output_two = run_schemas.RunOutput(
        type=run_schemas.RunIOType.stream,
        value=Stream(iter(["hello", "world"])),
        file=None,
    )
    static_output = run_schemas.RunOutput(
        type=run_schemas.RunIOType.string, value="static output", file=None
    )
    container_run_result = run_schemas.ContainerRunResult(
        inputs=None,
        outputs=[stream_output_one, static_output, stream_output_two],
        error=None,
    )

    results = [
        (result, status_code)
        async for result, status_code in _stream_run_outputs(
            container_run_result, DummyRequest()
        )
    ]
    data = [json.loads(result) for result, _ in results]
    status_codes = [status_code for _, status_code in results]
    # even if pipeline_error, status code should be 200
    assert all(status_code == 200 for status_code in status_codes)

    # exception was raised on 2nd iteration, so we expect there to be a valid
    # output followed by an error
    assert len(results) == 2

    assert data[0]["outputs"] == [
        {"type": "integer", "value": 1, "file": None},
        {"type": "string", "value": "static output", "file": None},
        {"type": "string", "value": "hello", "file": None},
    ]

    error = data[1].get("error")
    assert error is not None
    assert error["message"] == "Exception('dummy error')"
    assert error["type"] == "pipeline_error"
