import io
from typing import Any

from pipeline.api.call import post_file
from pipeline.schemas.file import FileGet
from pipeline.util import python_object_to_hex


def upload_file(file_or_path: Any, remote_path: str) -> FileGet:
    if isinstance(file_or_path, str):
        with open(file_or_path, "rb") as file:
            return post_file("/v2/files/", file, remote_path)
    else:
        return post_file("/v2/files/", file_or_path, remote_path)


def upload_python_object_to_file(obj: object, remote_path: str) -> FileGet:
    return upload_file(io.BytesIO(python_object_to_hex(obj).encode()), remote_path)
