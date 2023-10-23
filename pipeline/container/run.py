import asyncio
import logging

from fastapi import APIRouter, Request

from pipeline.cloud.schemas import runs as run_schemas
from pipeline.container.manager import Manager

logger = logging.getLogger("uvicorn")
router = APIRouter(prefix="/run")


@router.post(
    "",
    tags=["run"],
    status_code=200,
    # response_model=run_schemas.Run,
)
async def run(run_create: run_schemas.RunCreate, request: Request):
    outputs = await request.app.state.manager.run(run_create.input_data)

    return outputs


async def execution_handler(execution_queue: asyncio.Queue, manager: Manager) -> None:
    while True:
        try:
            args, response_queue, async_run = await execution_queue.get()

            output = await manager.run(*args)
            response_queue.put_nowait(output)
        except Exception:
            logger.exception("Got an error in the execution loop handler")
