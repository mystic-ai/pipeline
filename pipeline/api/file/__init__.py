import io

from pipeline.util import python_object_to_hex
from pipeline.api.call import post_file

from pipeline.schemas.file import FileGet


def upload_file(file_or_path, remote_path) -> FileGet:
    if isinstance(file_or_path, str):
        with open(file_or_path, "rb") as file:
            return post_file("/v2/files/", file, remote_path)
    else:
        return post_file("/v2/files/", file_or_path, remote_path)


def upload_python_object_to_file(obj, remote_path) -> FileGet:
    return upload_file(io.BytesIO(python_object_to_hex(obj).encode()), remote_path)
