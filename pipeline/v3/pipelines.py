import importlib
import math
import time
import typing as t
from multiprocessing import Pool
from pathlib import Path
from tempfile import SpooledTemporaryFile

import cloudpickle as cp
import httpx

from pipeline.objects import Graph, PipelineFile
from pipeline.util.logging import _print
from pipeline.v3 import http
from pipeline.v3.compute_requirements import Accelerator
from pipeline.v3.schemas import pipelines as pipeline_schemas
from pipeline.v3.schemas.runs import (
    Run,
    RunCreate,
    RunInput,
    RunIOType,
    RunOutput,
    RunState,
)


def upload_pipeline(
    graph: Graph,
    name: str,
    environment_id_or_name: t.Union[str, int],
    required_gpu_vram_mb: t.Optional[int] = None,
    minimum_cache_number: t.Optional[int] = None,
    modules: t.Optional[t.List[str]] = None,
    accelerators: t.Optional[t.List[Accelerator]] | None = None,
) -> pipeline_schemas.PipelineGet:
    if graph._has_run_startup:
        raise Exception("Graph has already been run, cannot upload")

    if modules is not None:
        for module_name in modules:
            module = importlib.import_module(module_name)
            cp.register_pickle_by_value(module)

    for variable in graph.variables:
        if isinstance(variable, PipelineFile):
            variable_path = Path(variable.path)
            if not variable_path.exists():
                raise FileNotFoundError(
                    f"File not found for variable (path={variable.path}, "
                    f"variable={variable.name})"
                )

            variable_file = variable_path.open("rb")
            try:
                res = http.post_files(
                    "/v3/pipeline_files",
                    files=dict(pfile=variable_file),
                    progress=True,
                )

                new_path = res.json()["path"]
                variable.path = new_path
                variable.remote_id = res.json()["id"]
            finally:
                variable_file.close()
    graph_file = SpooledTemporaryFile()

    graph_file.write(cp.dumps(graph))
    graph_file.seek(0)

    params = dict()

    params["name"] = name
    params["environment_id_or_name"] = environment_id_or_name

    if required_gpu_vram_mb is not None:
        params["gpu_memory_min"] = required_gpu_vram_mb
    if minimum_cache_number is not None:
        params["minimum_cache_number"] = minimum_cache_number

    if accelerators is not None:
        params["accelerators"] = [
            acc.value for acc in accelerators if isinstance(acc, Accelerator)
        ]

    res = http.post_files(
        "/v3/pipelines",
        files=dict(graph=graph_file),
        params=params,
    )

    if res.status_code != 200:
        print("Error uploading pipeline")
        print(res.text)
        res.raise_for_status()

    graph_file.close()

    raw_json = res.json()

    return pipeline_schemas.PipelineGet.parse_obj(raw_json)


def _data_to_run_input(data: t.Any) -> t.List[RunInput]:
    input_array = []

    for item in data:
        input_type = RunIOType.from_object(item)
        if input_type == RunIOType.file:
            raise NotImplementedError("File input not yet supported")
        elif input_type == RunIOType.pkl:
            raise NotImplementedError("Python object input not yet supported")

        input_schema = RunInput(
            type=input_type,
            value=item,
        )
        input_array.append(input_schema)

    return input_array


def run_pipeline(
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

    res = http.post(
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

    # Everything is okay!
    run_get = Run.parse_obj(res.json())

    return run_get


def get_pipeline_run(run_id: str) -> Run:
    http_res = http.get(f"/v3/runs/{run_id}")

    run_get = Run.parse_obj(http_res.json())
    return run_get


def map_pipeline_mp(array: list, graph_id: str, *, pool_size=8):
    results = []

    num_batches = math.ceil(len(array) / pool_size)
    for b in range(num_batches):
        start_index = b * pool_size
        end_index = min(len(array), start_index + pool_size)

        inputs = [[graph_id, item] for item in array[start_index:end_index]]

        with Pool(pool_size) as p:
            batch_res = p.starmap(run_pipeline, inputs)
            results.extend(batch_res)

    return results


def stream_pipeline(
    pipeline_id_or_pointer: str,
    *data,
) -> t.Iterator[RunOutput]:
    run_create_schema = RunCreate(
        pipeline_id_or_pointer=pipeline_id_or_pointer,
        input_data=_data_to_run_input(data),
        async_run=False,
    )
    with http.stream_post(
        "/v3/runs/stream",
        json_data=run_create_schema.dict(),
    ) as generator:
        for item in generator.iter_text():
            if item:
                yield RunOutput.parse_raw(item)


def poll_async_run(
    run_id: str, *, timeout: int = None, interval: float | int = 1.0
) -> Run:
    start_time = time.time()
    while True:
        run_get = get_pipeline_run(run_id)
        if run_get.state in [
            RunState.completed,
            RunState.failed,
            RunState.rate_limited,
            RunState.lost,
            RunState.no_environment_installed,
        ]:
            return run_get

        if timeout is not None and time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout waiting for run (run_id={run_id})")

        time.sleep(interval)
