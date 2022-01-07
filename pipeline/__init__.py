import os

from pipeline.api.cloud import PipelineCloud
from pipeline.objects import Pipeline, Variable, pipeline_function, pipeline_model

# from pipeline import logging

__all__ = [
    "Pipeline",
    "Variable",
    "pipeline_model",
    "pipeline_function",
    "PipelineCloud",
]

CACHE_DIR = os.getenv("PIPELINE_CACHE_DIR", "./cache")

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
elif not os.path.isdir(CACHE_DIR):
    raise Exception("Cache dir '%s' is not a valid dir." % CACHE_DIR)
