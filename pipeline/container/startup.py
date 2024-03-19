import asyncio
import os
import traceback
import uuid

import pkg_resources
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from pipeline.cloud.schemas import runs as run_schemas
from pipeline.container.logging import redirect_stdout, setup_logging
from pipeline.container.manager import Manager
from pipeline.container.routes import router
from pipeline.container.status import router as status_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="pipeline-container",
    )

    setup_logging()

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
    static_dir = pkg_resources.resource_filename(
        "pipeline", "container/frontend/static"
    )

    app.mount(
        "/static",
        StaticFiles(directory=static_dir),
        name="static",
    )

    return app


def setup_middlewares(app: FastAPI) -> None:
    @app.middleware("http")
    async def _(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-Id") or str(
            uuid.uuid4()
        )

        try:
            response = await call_next(request)
            response.headers["X-Request-Id"] = request.state.request_id
        except Exception as e:
            logger.exception(e)
            response = JSONResponse(
                status_code=500,
                content={
                    "error": repr(e),
                    "traceback": str(traceback.format_exc()),
                },
            )
            response.headers["X-Request-Id"] = request.state.request_id
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


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
    with redirect_stdout():
        await run_in_threadpool(manager.startup)

        while True:
            try:
                args, response_queue = await execution_queue.get()
                args: run_schemas.ContainerRunCreate
                input_data = args.inputs
                run_id = args.run_id
                with logger.contextualize(run_id=run_id):
                    try:
                        output = await run_in_threadpool(
                            manager.run, run_id=run_id, input_data=input_data
                        )
                    except Exception as e:
                        logger.exception("Exception raised during pipeline execution")
                        response_queue.put_nowait(e)
                        continue
                    response_queue.put_nowait(output)
            except Exception:
                logger.exception("Got an error in the execution loop handler")
