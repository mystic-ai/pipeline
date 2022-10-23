import os
import sys
from pathlib import Path

PIPELINE_CACHE = os.getenv(
    "PIPELINE_CACHE",
    os.path.join(os.getenv("APPDATA"), "/pipeline/environments")
    if (sys.platform == "win32" or sys.platform == "cygwin")
    else os.path.join(str(Path.home()), ".cache/pipeline/environments"),
)


if not os.path.exists(PIPELINE_CACHE):
    os.makedirs(PIPELINE_CACHE)
