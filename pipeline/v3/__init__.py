import httpx

import typing as t

import cloudpickle as cp

from pipeline.objects.graph import Graph


def run_pipeline(graph: Graph, data: t.Any):

    with open("graph.tmp", "wb") as tmp:
        tmp.write(cp.dumps(graph))

    with open("data.tmp", "wb") as tmp:
        tmp.write(cp.dumps(data))

    graph_file = open("graph.tmp", "rb")
    data_file = open("data.tmp", "rb")

    res = httpx.post(
        "http://localhost:5025/v3/runs",
        files=dict(graph=graph_file, input_data=data_file),
    )

    graph_file.close()
    data_file.close()

    return res


def upload_pipeline(graph: Graph):

    with open("graph.tmp", "wb") as tmp:
        tmp.write(cp.dumps(graph))

    graph_file = open("graph.tmp", "rb")

    res = httpx.post(
        "http://localhost:5025/v3/pipelines",
        files=dict(graph=graph_file),
    )

    graph_file.close()

    return res
