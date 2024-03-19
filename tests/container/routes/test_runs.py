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
    for result in results:
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
