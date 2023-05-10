import io
import math
import typing as t
from multiprocessing import Pool
from pathlib import Path

import cloudpickle as cp

from pipeline.objects import Graph, PipelineFile
from pipeline.util.logging import _print
from pipeline.v3 import http


def upload_pipeline(
    graph: Graph,
    gpu_memory_min: int = None,
    environment_id_or_name: t.Union[str, int] = None,
):
    if graph._has_run_startup:
        raise Exception("Graph has already been run, cannot upload")

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

    with open("graph.tmp", "wb") as tmp:
        tmp.write(cp.dumps(graph))

    graph_file = open("graph.tmp", "rb")

    params = dict()
    if gpu_memory_min is not None:
        params["gpu_memory_min"] = gpu_memory_min

    if isinstance(environment_id_or_name, int):
        params["environment_id"] = environment_id_or_name
    elif isinstance(environment_id_or_name, str):
        params["environment_name"] = environment_id_or_name

    res = http.post_files(
        "/v3/pipelines",
        files=dict(graph=graph_file),
        params=params,
    )

    graph_file.close()

    return res


def run_pipeline(graph_id: str, data: t.Any):
    data_obj = io.BytesIO(cp.dumps(data))

    res = http.post_files(
        "/v3/runs",
        params=dict(graph_id=graph_id),
        files=dict(input_data=data_obj),
    )

    if res.status_code != 200:
        _print(
            f"Failed run (status={res.status_code}, text={res.text}, "
            f"headers={res.headers})",
            level="ERROR",
        )
        raise Exception(f"Error: {res.status_code}, {res.text}")

    return res.json()["result"][0]


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
