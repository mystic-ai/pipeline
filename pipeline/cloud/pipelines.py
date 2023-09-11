import importlib
import json
import math
import os
import platform
import time
import typing as t
import uuid
from importlib.metadata import version
from multiprocessing import Pool
from pathlib import Path
from tempfile import SpooledTemporaryFile
from threading import Thread
from urllib.parse import urlparse
from zipfile import ZipFile

import cloudpickle as cp
import httpx
from httpx import HTTPStatusError
from pydantic import ValidationError

from pipeline.cloud import http
from pipeline.cloud.compute_requirements import Accelerator
from pipeline.cloud.logs import tail_run_logs
from pipeline.cloud.schemas import pipelines as pipeline_schemas
from pipeline.cloud.schemas.runs import (
    Run,
    RunCreate,
    RunError,
    RunInput,
    RunIOType,
    RunOutput,
    RunState,
)
from pipeline.configuration import current_configuration
from pipeline.objects import Directory, File, FileURL, Graph
from pipeline.util.logging import _print, _print_remote_log


class PipelineRunError(Exception):
    """Error raised when there was an exception raised during a pipeline run"""

    def __init__(
        self,
        exception: str,
        traceback: t.Optional[str],
    ) -> None:
        self.exception = exception
        self.traceback = traceback
        super().__init__(self.exception, self.traceback)

    def __str__(self):
        error_msg = self.exception
        if self.traceback:
            error_msg += f"\n\nFull traceback from run:\n\n{self.traceback}"
        return error_msg


def upload_pipeline(
    graph: Graph,
    name: str,
    environment_id_or_name: t.Union[str, int],
    required_gpu_vram_mb: t.Optional[int] = None,
    minimum_cache_number: t.Optional[int] = None,
    modules: t.Optional[t.List[str]] = None,
    accelerators: t.Optional[t.List[Accelerator]] | None = None,
    _metadata: t.Optional[dict] = None,
) -> pipeline_schemas.PipelineGet:
    if graph._has_run_startup:
        raise Exception("Graph has already been run, cannot upload")

    if modules is not None:
        for module_name in modules:
            module = importlib.import_module(module_name)
            cp.register_pickle_by_value(module)

    for variable in graph.variables:
        if isinstance(variable, Directory):
            if variable.remote_id is not None:
                continue
            variable_path = Path(variable.path)

            if not variable_path.exists():
                raise FileNotFoundError(
                    f"Directory not found for variable (path={variable.path}) "
                )
            if not variable_path.is_dir() and not str(variable_path).endswith(".zip"):
                raise ValueError(
                    f"Variable is not a directory or zip file (path={variable.path})"
                )

            zip_path = variable_path

            if not str(variable_path).endswith(".zip"):
                tmp_path = Path("/tmp") / (str(uuid.uuid4()) + ".zip")
                with ZipFile(str(tmp_path), "w") as zip_file:
                    for root, dirs, files in os.walk(str(variable_path)):
                        for file in files:
                            zip_file.write(
                                os.path.join(root, file),
                            )
                zip_path = tmp_path

            zip_file = zip_path.open("rb")
            try:
                res = http.post_files(
                    "/v3/pipeline_files",
                    files=dict(pfile=zip_file),
                    progress=True,
                )

                new_path = res.json()["path"]
                variable.path = Path(new_path)
                variable.remote_id = res.json()["id"]
            except HTTPStatusError as e:
                if e.response.status_code == 403:
                    raise Exception(
                        f"Permission denied uploading file (path={variable.path}, "
                        f"variable={variable.title})"
                    )
            finally:
                zip_file.close()
        elif isinstance(variable, FileURL):
            try:
                urlparse(variable.url)
            except Exception:
                raise Exception(
                    f"Invalid URL for variable (url={variable.url})",
                )
        elif isinstance(variable, File):
            if variable.remote_id is not None:
                continue
            variable_path = Path(variable.path)

            if not variable_path.exists():
                raise FileNotFoundError(
                    f"File not found for variable (path={variable.path}, "
                    f"variable={variable.title})"
                )

            variable_file = variable_path.open("rb")
            try:
                res = http.post_files(
                    "/v3/pipeline_files",
                    files=dict(pfile=variable_file),
                    progress=True,
                )

                new_path = res.json()["path"]
                variable.path = Path(new_path)
                variable.remote_id = res.json()["id"]
            except HTTPStatusError as e:
                if e.response.status_code == 403:
                    raise Exception(
                        f"Permission denied uploading file (path={variable.path}, "
                        f"variable={variable.title})"
                    )
            finally:
                variable_file.close()

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

    input_variables: t.List[dict] = []
    output_variables: t.List[dict] = []

    for variable in graph.variables:
        if variable.is_input:
            input_variables.append(variable.to_io_schema().json())

        if variable.is_output:
            output_variables.append(variable.to_io_schema().json())

    params["input_variables"] = input_variables
    params["output_variables"] = output_variables

    if _metadata is None:
        _metadata = dict()

    default_meta = dict(
        python_version=platform.python_version(),
        system=platform.system(),
        platform_version=platform.version(),
        platform=platform.platform(),
        sdk_version=version("pipeline-ai"),
    )

    _metadata.update(default_meta)

    params["_metadata"] = json.dumps(_metadata)

    graph_file = SpooledTemporaryFile()
    graph_file.write(cp.dumps(graph))
    graph_file.seek(0)
    try:
        res = http.post_files(
            "/v3/pipelines",
            files=dict(graph=graph_file),
            data=params,
        )
    except HTTPStatusError as e:
        if e.response.status_code == 403:
            raise Exception(
                e.response.text,
            )
        elif e.response.status_code == 400:
            raise Exception(
                e.response.text,
            )
        elif e.response.status_code == 409:
            raise Exception(
                e.response.text,
            )
        else:
            raise Exception(
                f"Unknown exception (code={e.response.status_code}), {e.response.text}"
            )

    if res.status_code != 200:
        print("Error uploading pipeline")
        print(res.text)
        res.raise_for_status()

    graph_file.close()

    raw_json = res.json()

    try:
        pointers_request = http.get(
            "/v3/pointers",
            params=dict(
                pipeline_name=name,
            ),
        )
        pointer_array = pointers_request.json()["data"]
        for item in pointer_array:
            if item["pipeline_id"] == raw_json["id"]:
                name = item["pointer"]

                break
    except HTTPStatusError:
        ...

    _print(f"Uploaded pipeline '{name}' with ID = {raw_json['id']}", level="SUCCESS")
    return pipeline_schemas.PipelineGet.parse_obj(raw_json)


def _data_to_run_input(data: t.Tuple) -> t.List[RunInput]:
    input_array = []

    for item in data:
        input_type = RunIOType.from_object(item)
        if input_type == RunIOType.file or isinstance(item, File):
            if isinstance(item, FileURL):
                input_schema = RunInput(
                    value=None,
                    type=RunIOType.file,
                    file_path=item.url,
                    file_name=item.url.split("/")[-1],
                )
                input_array.append(input_schema)
            else:
                file_obj = open(item.path, "rb") if isinstance(item, File) else item

                res = http.post_files(
                    "/v3/pipeline_files",
                    files=dict(pfile=file_obj),
                    progress=True,
                )
                file_path = res.json().get("path", None)

                if file_path is None:
                    raise Exception("Error uploading file")

                input_schema = RunInput(
                    value=None,
                    type=input_type,
                    file_path=file_path,
                    file_name=file_path.split("/")[-1],
                )
                input_array.append(input_schema)
            continue
        elif input_type == RunIOType.pkl:
            raise NotImplementedError("Python object input not yet supported")

        input_schema = RunInput(
            type=input_type,
            value=item,
            file_path=None,
            file_name=None,
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
        async_run=async_run if not current_configuration.is_debugging() else True,
    )

    res = http.post(
        "/v3/runs",
        json_data=run_create_schema.dict(),
        raise_for_status=False,
    )

    if return_response and not current_configuration.is_debugging():
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

    def _print_logs():
        for log in tail_run_logs(run_get.id):
            _print_remote_log(log)
            # _print(log)

    if current_configuration.is_debugging():
        log_thread = Thread(target=_print_logs)

        _print("Running in debug mode")
        _print(f"Run ID: {run_get.id}")

        def activate_log_thread(state: RunState):
            if state == RunState.running:
                log_thread.start()

        run_get = poll_async_run(
            run_get.id,
            interval=0.5,
            state_change_callback=activate_log_thread,
        )

        if run_get.state == RunState.completed:
            _print("Run completed successfully", level="SUCCESS")
            if run_get.result is not None:
                _print(f"Result array: {run_get.result.result_array()}")

    if RunState.is_terminal(run_get.state) and run_get.state != RunState.completed:
        error = getattr(run_get, "error", None)

        traceback = getattr(error, "traceback", None)
        exception = getattr(error, "exception", None)
        if (
            isinstance(exception, str)
            and "Tried to return a non-supported type from a pipeline" in exception
        ):
            raise Exception(
                f"Remote Run '{run_get.id}' failed because an unsupported type was returned. Check your outputs.\n(More information -> https://docs.mystic.ai/docs/inputoutpu-types)"  # noqa
            )

        raise Exception(
            f"Remote Run '{run_get.id}' failed with exception: "
            f"{traceback if traceback is not None and traceback != '' else exception}"
        )

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
        run_error = None
        for item in generator.iter_text():
            if item:
                try:
                    output = RunOutput.parse_raw(item)
                    yield output
                except ValidationError:
                    run_error = RunError.parse_raw(item)
                if run_error is not None:
                    raise PipelineRunError(
                        run_error.exception, traceback=run_error.traceback
                    )


def poll_async_run(
    run_id: str,
    *,
    timeout: int | None = None,
    interval: float | int = 1.0,
    state_change_callback: t.Callable[[RunState], None] | None = None,
) -> Run:
    start_time = time.time()
    current_state = None
    while True:
        run_get = get_pipeline_run(run_id)
        if current_state != run_get.state and current_configuration.is_debugging():
            _print(f"Run state: {run_get.state.name}")
            current_state = run_get.state
            if state_change_callback is not None:
                state_change_callback(run_get.state)
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
