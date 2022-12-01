import base64
import json
import os
import sys
from pathlib import Path
from platform import python_version
from typing import TypedDict

from packaging import version

from pipeline.util.logging import _print

PIPELINE_CACHE = Path(
    os.getenv(
        "PIPELINE_CACHE",
        Path(os.getenv("LOCALAPPDATA")) / ".pipeline/"
        if (sys.platform == "win32" or sys.platform == "cygwin")
        else Path.home() / ".cache/pipeline/",
    )
)

PIPELINE_CACHE_AUTH = PIPELINE_CACHE / "auth.json"

if version.parse(python_version()) < version.parse("3.9.13"):
    _print(
        f"You are using python version '{python_version()}' "
        "please upgrade to python >=3.9.13 to ensure correct serialisation.",
        "WARNING",
    )

remote_auth: TypedDict("remote_auth", {"url": str}) = dict()


def _load_auth():
    if PIPELINE_CACHE_AUTH.exists():
        remote_auth = json.loads(PIPELINE_CACHE_AUTH.read_text())
        remote_auth = {
            url: base64.b64decode(encoded_token).decode()
            for url, encoded_token in remote_auth.items()
        }


def _save_auth():
    PIPELINE_CACHE.mkdir(exist_ok=True)
    with open(os.path.join(PIPELINE_CACHE, "auth.json"), "w") as auth_file:

        _b64_remote_auth = {
            auth[0]: base64.b64encode(auth[1].encode()).decode()
            for auth in remote_auth.items()
        }

        auth_file.write(json.dumps(_b64_remote_auth))


if PIPELINE_CACHE_AUTH.exists():
    _load_auth()
