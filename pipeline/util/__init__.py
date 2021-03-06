import random
import string
from typing import Any, Optional

from cloudpickle import dumps, loads

from pipeline.schemas.file import FileCreate


def generate_id(length: int) -> str:
    return "".join((random.choice(string.ascii_letters) for i in range(length)))


def python_object_to_hex(obj: Any) -> str:
    return dumps(obj).hex()


def hex_to_python_object(hex: str) -> Any:
    h = bytes.fromhex(hex).replace(b"\r\n", b"\n")
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
