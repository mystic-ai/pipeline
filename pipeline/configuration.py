import base64
import json
import os
import sys
from pathlib import Path
from platform import python_version
from typing import TypedDict

from packaging import version

from pipeline.util.logging import _print

PIPELINE_CACHE = os.getenv(
    "PIPELINE_CACHE",
    os.path.join(os.getenv("APPDATA"), "/pipeline/")
    if (sys.platform == "win32" or sys.platform == "cygwin")
    else os.path.join(str(Path.home()), ".cache/pipeline/"),
)

if not os.path.exists(PIPELINE_CACHE):
    os.makedirs(PIPELINE_CACHE)

if version.parse(python_version()) < version.parse("3.9.13"):
    _print(
        f"You are using python version '{python_version()}' "
        "please upgrade to python >=3.9.13 to ensure correct serialisation.",
        "WARNING",
    )

remote_auth: TypedDict("remote_auth", {"url": str}) = dict()

if os.path.exists(os.path.join(PIPELINE_CACHE, "auth.json")):
    with open(os.path.join(PIPELINE_CACHE, "auth.json"), "r") as auth_file:
        remote_auth = json.loads(auth_file.read())
        remote_auth = {
            auth[0]: base64.b64decode(auth[1]).decode() for auth in remote_auth.items()
        }


def _load_auth():
    if os.path.exists(os.path.join(PIPELINE_CACHE, "auth.json")):
        with open(os.path.join(PIPELINE_CACHE, "auth.json"), "r") as auth_file:
            remote_auth = json.loads(auth_file.read())
            remote_auth = {
                auth[0]: base64.b64decode(auth[1]).decode()
                for auth in remote_auth.items()
            }
    else:
        raise Exception("Authentication file not found")


def _save_auth():
    with open(os.path.join(PIPELINE_CACHE, "auth.json"), "w") as auth_file:

        _b64_remote_auth = {
            auth[0]: base64.b64encode(auth[1].encode()).decode()
            for auth in remote_auth.items()
        }

        auth_file.write(json.dumps(_b64_remote_auth))
