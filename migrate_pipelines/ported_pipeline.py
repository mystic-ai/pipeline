# flake8: noqa
import cloudpickle

with open(
    "./pipeline.graph",
    "rb",
) as f:
    graph = f.read()

ported_pipeline = cloudpickle.loads(graph)
