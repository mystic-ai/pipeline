import os

from pipeline.objects.pipeline_v2 import PipelineV2

__all__ = ["PipelineV2"]

CACHE_HOME = os.getenv("XDG_CACHE_HOME", "HOME")
CACHE_DIR = os.path.join(CACHE_HOME, ".cache")

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
elif not os.path.isdir(CACHE_DIR):
    raise Exception("Cache dir '%s' is not a valid dir." % CACHE_DIR)
