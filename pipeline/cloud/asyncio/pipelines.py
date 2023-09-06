import typing as t

import httpx

from pipeline.cloud import http
from pipeline.cloud.pipelines import _data_to_run_input
from pipeline.cloud.schemas.runs import Run, RunCreate, RunState
from pipeline.util.logging import _print


async def run_pipeline(
    pipeline_id_or_pointer: t.Union[str, int],
    *data,
    async_run: bool = False,
    return_response: bool = False,
) -> t.Union[Run, httpx.Response]:
    run_create_schema = RunCreate(
        pipeline_id_or_pointer=pipeline_id_or_pointer,
        input_data=_data_to_run_input(data),
        async_run=async_run,
    )

    res = await http.async_post(
        "/v3/runs",
        json_data=run_create_schema.dict(),
        raise_for_status=False,
    )

    if return_response:
        return res

    if res.status_code == 500:
        _print(
            f"Failed run (status={res.status_code}, text={res.text}, "
            f"headers={res.headers})",
            level="ERROR",
        )
        raise Exception(f"Error: {res.status_code}, {res.text}", res.status_code)
    elif res.status_code == 429:
        _print(
            f"Too many requests (status={res.status_code}, text={res.text})",
            level="ERROR",
        )
        raise Exception(
            "Too many requests, please try again later",
            res.status_code,
        )
    elif res.status_code == 404:
        _print(
            f"Pipeline not found (status={res.status_code}, text={res.text})",
            level="ERROR",
        )
        raise Exception("Pipeline not found", res.status_code)
    elif res.status_code == 503:
        _print(
            f"Environment not cached (status={res.status_code}, text={res.text})",
            level="ERROR",
        )
        raise Exception("Environment not cached", res.status_code)
    elif res.status_code == 502:
        _print(
            "Gateway error",
            level="ERROR",
        )
        raise Exception("Gateway error", res.status_code)

    run_get = Run.parse_obj(res.json())

    if RunState.is_terminal(run_get.state) and run_get.state != RunState.completed:
        raise Exception(
            f"Remote Run '{run_get.id}' failed with exception: "
            f"{getattr(getattr(run_get, 'error', None), 'traceback', None)}"
        )

    return run_get
