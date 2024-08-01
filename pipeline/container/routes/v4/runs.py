import asyncio
import traceback

from fastapi import APIRouter, Request, Response, status
from loguru import logger

from pipeline.cloud.http import StreamingResponseWithStatusCode
from pipeline.cloud.schemas import pipelines as pipeline_schemas
from pipeline.cloud.schemas import runs as run_schemas
from pipeline.container.manager import Manager
from pipeline.exceptions import RunnableError

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post(
    "",
    status_code=200,
    responses={
        500: {
            "description": "Pipeline failed",
            "model": run_schemas.ContainerRunResult,
        },
        400: {
            "description": "Invalid input data",
            "model": run_schemas.ContainerRunResult,
        },
        202: {
            "description": "Async run initiated",
            "model": run_schemas.ContainerRunResult,
        },
    },
)
async def run(
    run_create: run_schemas.ContainerRunCreate,
    request: Request,
    response: Response,
) -> run_schemas.ContainerRunResult:
    """Run this pipeline with the given inputs.

    If `async_run=True` then this route will return an empty result immediately,
    then make a POST call to the provided `callback_url` once the run is
    complete.
    """
    run_id = run_create.run_id
    with logger.contextualize(run_id=run_id):
        logger.info(f"Received run request; async_run={run_create.async_run}")
        manager: Manager = request.app.state.manager
        if result := _handle_pipeline_state_not_ready(manager):
            return result

        execution_queue: asyncio.Queue = request.app.state.execution_queue
        # If async run, we put run on the queue and return immediately
        if run_create.async_run is True:
            # check request is valid
            if not run_create.callback_url:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return run_schemas.ContainerRunResult(
                    outputs=None,
                    inputs=None,
                    error=run_schemas.ContainerRunError(
                        type=run_schemas.ContainerRunErrorType.input_error,
                        message="callback_url is required for async runs",
                        traceback=None,
                    ),
                )

            execution_queue.put_nowait((run_create, None))
            # return empty result for now with a status code of 202 to indicate
            # we have accepted the request and are processing it in the
            # background
            response.status_code = status.HTTP_202_ACCEPTED
            return run_schemas.ContainerRunResult(
                outputs=None,
                error=None,
                inputs=None,
            )
        # Otherwise, we put run on the queue then wait for the run to finish and
        # return the result
        response_queue: asyncio.Queue = asyncio.Queue()
        execution_queue.put_nowait((run_create, response_queue))

        response_schema, response.status_code = await response_queue.get()
        logger.info("Returning run result")
        return response_schema


@router.post(
    "/stream",
    status_code=200,
    responses={
        500: {
            "description": "Pipeline failed",
            "model": run_schemas.ContainerRunResult,
        },
        400: {
            "description": "Invalid input data",
            "model": run_schemas.ContainerRunResult,
        },
    },
)
async def stream_run(
    run_create: run_schemas.ContainerRunCreate,
    request: Request,
    response: Response,
):
    run_id = run_create.run_id
    with logger.contextualize(run_id=run_id):
        manager: Manager = request.app.state.manager
        if result := _handle_pipeline_state_not_ready(manager):
            return result

        execution_queue: asyncio.Queue = request.app.state.execution_queue

        response_queue: asyncio.Queue = asyncio.Queue()
        execution_queue.put_nowait((run_create, response_queue))
        # wait for result
        response_schema, response.status_code = await response_queue.get()

        outputs = response_schema.outputs or []
        if not outputs:
            return response_schema

        if not any([output.type == run_schemas.RunIOType.stream for output in outputs]):
            raise TypeError("No streaming outputs found")

        return StreamingResponseWithStatusCode(
            _stream_run_outputs(response_schema, request),
            media_type="application/json",
            # hint to disable buffering
            headers={"X-Accel-Buffering": "no"},
        )


def _fetch_next_outputs(outputs: list[run_schemas.RunOutput]):
    next_outputs = []
    have_new_streamed_outputs = False
    for output in outputs:
        if output.type == run_schemas.RunIOType.stream:
            if output.value is None:
                raise Exception("Stream value was None")

            try:
                next_value = output.value.__next__()
                next_outputs.append(
                    run_schemas.RunOutput(
                        type=run_schemas.RunIOType.from_object(next_value),
                        value=next_value,
                        file=None,
                    )
                )
                have_new_streamed_outputs = True
            except StopIteration:
                # if no data left for this stream, return None value
                next_outputs.append(
                    run_schemas.RunOutput(
                        type=run_schemas.RunIOType.none,
                        value=None,
                        file=None,
                    )
                )
            except Exception as exc:
                logger.exception("Pipeline error caught during run streaming")
                raise RunnableError(exception=exc, traceback=traceback.format_exc())
        else:
            next_outputs.append(output)

    if not have_new_streamed_outputs:
        return
    return next_outputs


async def _stream_run_outputs(
    response_schema: run_schemas.ContainerRunResult, request: Request
):
    """Generator returning output data for list of outputs.

    We iterate over all outputs until we no longer have any streamed data to
    output
    """
    outputs = response_schema.outputs or []
    while True:
        status_code = 200
        try:
            next_outputs = _fetch_next_outputs(outputs)
            if not next_outputs:
                return

            new_response_schema = run_schemas.ContainerRunResult(
                inputs=response_schema.inputs,
                outputs=next_outputs,
                error=response_schema.error,
            )

        except RunnableError as e:
            # if we get a pipeline error, return a run error then finish
            new_response_schema = run_schemas.ContainerRunResult(
                outputs=None,
                inputs=response_schema.inputs,
                error=run_schemas.ContainerRunError(
                    type=run_schemas.ContainerRunErrorType.pipeline_error,
                    message=repr(e.exception),
                    traceback=e.traceback,
                ),
            )
        except Exception as e:
            logger.exception("Unexpected error during run streaming")
            status_code = 500
            new_response_schema = run_schemas.ContainerRunResult(
                outputs=None,
                inputs=response_schema.inputs,
                error=run_schemas.ContainerRunError(
                    type=run_schemas.ContainerRunErrorType.unknown,
                    message=repr(e),
                    traceback=None,
                ),
            )

        # serialise response to str and add newline separator
        yield f"{new_response_schema.json()}\n", status_code

        # if there was an error or request is disconnected terminate all iterators
        if new_response_schema.error or await request.is_disconnected():
            for output in outputs:
                if (
                    output.type == run_schemas.RunIOType.stream
                    and output.value is not None
                    and hasattr(output.value.iterable, "end")
                ):
                    output.value.iterable.end()
            return


def _handle_pipeline_state_not_ready(
    manager: Manager,
) -> run_schemas.ContainerRunResult | None:
    if manager.pipeline_state == pipeline_schemas.PipelineState.loading:
        logger.info("Pipeline loading")
        return run_schemas.ContainerRunResult(
            outputs=None,
            inputs=None,
            error=run_schemas.ContainerRunError(
                type=run_schemas.ContainerRunErrorType.pipeline_loading,
                message="Pipeline is still loading",
                traceback=None,
            ),
        )

    if manager.pipeline_state == pipeline_schemas.PipelineState.load_failed:
        logger.info("Pipeline failed to load")
        return run_schemas.ContainerRunResult(
            outputs=None,
            inputs=None,
            error=run_schemas.ContainerRunError(
                type=run_schemas.ContainerRunErrorType.startup_error,
                message="Pipeline failed to load",
                traceback=manager.pipeline_state_message,
            ),
        )

    if manager.pipeline_state == pipeline_schemas.PipelineState.startup_failed:
        logger.info("Pipeline failed to startup")
        return run_schemas.ContainerRunResult(
            outputs=None,
            inputs=None,
            error=run_schemas.ContainerRunError(
                type=run_schemas.ContainerRunErrorType.startup_error,
                message="Pipeline failed to startup",
                traceback=manager.pipeline_state_message,
            ),
        )
