import typing as t

import httpx

from pipeline.cloud import http
from pipeline.cloud.pipelines import _data_to_run_input
from pipeline.cloud.schemas.runs import Run, RunCreate, RunState


async def run_pipeline(
    pipeline: str,
    *data,
    async_run: bool = False,
    return_response: bool = False,
) -> t.Union[Run, httpx.Response]:
    run_create_schema = RunCreate(
        pipeline=pipeline,
        inputs=_data_to_run_input(data),
        async_run=async_run,
    )

    res = await http.async_post(
        "/v4/runs",
        json_data=run_create_schema.dict(),
        handle_error=not return_response,
    )

    if return_response:
        return res

    run_get = Run.parse_obj(res.json())

    if RunState.is_terminal(run_get.state) and run_get.state != RunState.completed:
        raise Exception(
            f"Remote Run '{run_get.id}' failed with exception: "
            f"{getattr(getattr(run_get, 'error', None), 'traceback', None)}"
        )

    return run_get
