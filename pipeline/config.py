import os
import sys
from pathlib import Path
from platform import python_version

from packaging import version

from pipeline.util.logging import _print

PIPELINE_CACHE = os.getenv(
    "PIPELINE_CACHE",
    os.path.join(os.getenv("APPDATA"), "/pipeline/environments")
    if (sys.platform == "win32" or sys.platform == "cygwin")
    else os.path.join(str(Path.home()), ".cache/pipeline/environments"),
)


if not os.path.exists(PIPELINE_CACHE):
    os.makedirs(PIPELINE_CACHE)

if version.parse(python_version()) < version.parse("3.9.13"):
    _print(
        f"You are using python version '{python_version()}' "
        "please upgrade to python >=3.9.13 to ensure correct serialisation.",
        "WARNING"
    )
