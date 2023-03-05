import asyncio
import io
import math
import time
import typing as t
from multiprocessing import Pool

import cloudpickle as cp
import httpx
import requests

from pipeline.objects.graph import Graph

LOCAL_IP = "10.1.255.139:5025"
TESTING_API_IP = "10.1.255.156:5025"

ACTIVE_IP = LOCAL_IP


def run_pipeline(graph_id: str, data: t.Any):
    data_obj = io.BytesIO(cp.dumps(data))
    start_time = time.time()

    # headers = {
    #     "Content-ype": "multipart/form-data",
    # }

    res = requests.post(
        f"http://{ACTIVE_IP}/v3/runs",
        params=dict(graph_id=graph_id),
        files=dict(input_data=data_obj),
        # headers=headers,
    )
    end_time = time.time()
    print(f"Run time: {end_time - start_time}")

    return res.json()["result"][0]


def upload_pipeline(graph: Graph, gpu_memory_min: int = None):
    with open("graph.tmp", "wb") as tmp:
        tmp.write(cp.dumps(graph))

    graph_file = open("graph.tmp", "rb")

    params = dict()
    if gpu_memory_min is not None:
        params["gpu_memory_min"] = gpu_memory_min

    res = httpx.post(
        f"http://{ACTIVE_IP}/v3/pipelines",
        files=dict(graph=graph_file),
        params=params,
    )

    graph_file.close()

    return res


def map_pipeline(array: list, graph_id: str):
    async def run(data: t.Any):
        async with httpx.AsyncClient() as client:
            obj_bytes = io.BytesIO(cp.dumps(data))

            r = await client.post(
                f"http://{ACTIVE_IP}/v3/runs",
                params=dict(graph_id=graph_id),
                files=dict(input_data=obj_bytes),
            )
            return r.json()["result"][0]

    reqs = []
    for item in array:
        reqs.append(run(item))

    async def main():
        return await asyncio.gather(*reqs)

    return asyncio.run(main())


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
