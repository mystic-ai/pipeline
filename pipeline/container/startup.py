import asyncio
import logging
import os
import threading
import time
import traceback
import uuid

from fastapi import FastAPI, Request
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from pipeline.cloud.schemas import pipelines as pipeline_schemas
from pipeline.container.manager import Manager
from pipeline.container.routes import router
from pipeline.container.status import router as status_router

logger = logging.getLogger("uvicorn")


def create_app() -> FastAPI:
    app = FastAPI(
        title="pipeline-container",
    )

    setup_oapi(app)
    setup_middlewares(app)

    app.state.execution_queue = asyncio.Queue()
    app.state.manager = Manager(
        pipeline_path=os.environ.get(
            "PIPELINE_PATH",
            "",
        )
    )
    asyncio.create_task(execution_handler(app.state.execution_queue, app.state.manager))

    app.include_router(router)
    app.include_router(status_router)
    app.mount(
        "/static",
        StaticFiles(directory="/app/container/frontend/static"),
        name="static",
    )

    return app


def setup_middlewares(app: FastAPI) -> None:
    @app.middleware("http")
    async def _(request: Request, call_next):
        try:
            response = await call_next(request)
        except Exception as e:
            logger.exception(e)
            return JSONResponse(
                status_code=500,
                content={
                    "traceback": str(traceback.format_exc()),
                },
            )
        return response

    @app.middleware("http")
    async def _(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-Id") or str(
            uuid.uuid4()
        )
        response = await call_next(request)
        response.headers["X-Request-Id"] = request.state.request_id
        return response


def setup_oapi(app: FastAPI) -> None:
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title="pipeline-container",
            version="1.1.0",
            routes=app.routes,
            servers=[{"url": "http://localhost:14300"}],
        )
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi


async def execution_handler(execution_queue: asyncio.Queue, manager: Manager) -> None:
    threading.Thread(target=asyncio.run, args=(manager.startup(),)).start()

    while True:
        try:
            args, response_queue = await execution_queue.get()

            start_time = time.time()
            timedout = False

            logger.info("Got run request")
            logger.info(f"Pipeline state: {manager.pipeline_state}")

            while manager.pipeline_state == pipeline_schemas.PipelineState.loading:
                if time.time() - start_time > 30:
                    timedout = True
                    break

                await asyncio.sleep(0.1)

            if timedout:
                response_queue.put_nowait(Exception())
                logger.info("Loading timedout")
                continue

            if manager.pipeline_state == pipeline_schemas.PipelineState.failed:
                response_queue.put_nowait(Exception())
                logger.info("Pipeline failed to load")
                continue

            try:
                output = await manager.run(args)
            except Exception as e:
                logger.exception(e)
                response_queue.put_nowait(e)
                continue
            response_queue.put_nowait(output)
        except Exception:
            logger.exception("Got an error in the execution loop handler")
