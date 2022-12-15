import io
import random
import string
from typing import Any, Callable, Optional

from cloudpickle import dumps
from dill import loads

from pipeline.schemas.file import FileCreate


def generate_id(length: int) -> str:
    return "".join((random.choice(string.ascii_letters) for i in range(length)))


def python_object_to_hex(obj: Any) -> str:
    return dumps(obj).hex()


def hex_to_python_object(hex: str) -> Any:
    h = bytes.fromhex(hex)
    return loads(h)


def python_object_to_name(obj: Any) -> Optional[str]:
    try:
        name = obj.__name__
    except Exception:
        name = None
    return name


def python_object_to_file_create(obj: Any, name: str = None):
    if name is None:
        name = generate_id(20)

    return FileCreate(name=name, file_bytes=python_object_to_hex)


class CallbackBytesIO(io.BytesIO):
    """Provides same interface as BytesIO but additionally calls a callback function
    whenever the 'read' method is called.

    This is similar to tqdm's own CallbackIOWrapper but this does not play nicely with
    all features of httpx so we use our own in some cases.
    """

    def __init__(self, callback: Callable, initial_bytes: bytes):
        self._callback = callback
        super().__init__(initial_bytes)

    def read(self, size=-1) -> bytes:
        data = super().read(size)
        self._callback(len(data))
        return data
