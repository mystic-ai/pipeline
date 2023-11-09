import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, Request, Response

from pipeline.cloud.schemas import runs as run_schemas
from pipeline.container.redis import redis_client
from pipeline.exceptions import RunInputException

logger = logging.getLogger("uvicorn")
router = APIRouter(prefix="/runs")


async def _wait_and_save_result(run_id, response_queue):
    run_output = await response_queue.get()
    if isinstance(run_output, RunInputException):
        # response.status_code = 400
        response_schema = run_schemas.ContainerRunResult(
            id=run_id,
            outputs=None,
            error=run_schemas.ContainerRunError.input_error,
            error_message=run_output.message,
        )

    elif isinstance(run_output, Exception):
        # response.status_code = 500
        response_schema = run_schemas.ContainerRunResult(
            id=run_id,
            outputs=None,
            error=run_schemas.ContainerRunError.unknown,
            error_message=str(run_output),
        )
    else:
        outputs = [
            run_schemas.RunOutput(
                type=run_schemas.RunIOType.from_object(output),
                value=output,
                file=None,
            )
            for output in run_output
        ]
        response_schema = run_schemas.ContainerRunResult(
            id=run_id,
            outputs=outputs,
            error=None,
            error_message=None,
        )

    redis_key = f"result:{run_id}"
    redis_client.set(redis_key, response_schema.json(), ex=60 * 60)
    # publish fact that result is ready
    redis_client.publish(redis_key, 1)
    return response_schema


@router.post(
    "",
    tags=["runs"],
    status_code=200,
    # response_model=run_schemas.Run,
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
async def run(
    run_create: run_schemas.ContainerRunCreate,
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
):
    # run_manager: Manager = request.app.state.manager
    # outputs = await run_manager.run(run_create.inputs)
    # return outputs

    execution_queue: asyncio.Queue = request.app.state.execution_queue

    response_queue: asyncio.Queue = asyncio.Queue()
    execution_queue.put_nowait((run_create.inputs, response_queue))
    background_tasks.add_task(_wait_and_save_result, run_create.id, response_queue)
    return {"status": "ok"}
