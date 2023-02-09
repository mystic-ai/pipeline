import asyncio
import io
import math
import typing as t
from multiprocessing import Pool

import cloudpickle as cp
import httpx
import requests

from pipeline.objects.graph import Graph


def run_pipeline(graph_id: str, data: t.Any):

    # with open("graph.tmp", "wb") as tmp:
    #     tmp.write(cp.dumps(graph))

    with open("data.tmp", "wb") as tmp:
        tmp.write(cp.dumps(data))

    # graph_file = open("graph.tmp", "rb")
    data_file = open("data.tmp", "rb")

    res = requests.post(
        "http://10.1.255.139:5025/v3/runs",
        params=dict(graph_id=graph_id),
        files=dict(input_data=data_file),
    )

    # graph_file.close()
    data_file.close()

    # return res.json()["result"][0]
    return res


def upload_pipeline(graph: Graph):

    with open("graph.tmp", "wb") as tmp:
        tmp.write(cp.dumps(graph))

    graph_file = open("graph.tmp", "rb")

    res = httpx.post(
        "http://10.1.255.139:5025/v3/pipelines",
        files=dict(graph=graph_file),
    )

    graph_file.close()

    return res


def map_pipeline(array: list, graph_id: str):
    async def run(data: t.Any):
        async with httpx.AsyncClient() as client:
            obj_bytes = io.BytesIO(cp.dumps(data))

            r = await client.post(
                "http://10.1.255.139:5025/v3/runs",
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
