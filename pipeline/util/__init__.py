import importlib
import importlib.metadata
import io
import random
import string
from typing import Any, Callable, List, Optional, Union

from cloudpickle import dumps, register_pickle_by_value
from dill import loads

from pipeline.schemas.file import FileCreate


def package_version() -> str:
    """Return the version of the installed `pipeline-ai` package."""
    return importlib.metadata.version("pipeline-ai")


def generate_id(length: int) -> str:
    return "".join((random.choice(string.ascii_letters) for i in range(length)))


def python_object_to_hex(
    obj: Any,
    modules: Optional[List[str]] = None,
) -> str:
    return dump_object(obj, modules=modules).hex()


def hex_to_python_object(hex: str) -> Any:
    h = bytes.fromhex(hex)
    return loads(h)


def dump_object(obj: Any, modules: Optional[List[str]] = None) -> bytes:
    """Serialize `obj` as bytes."""
    if modules is not None:
        for module_name in modules:
            module = importlib.import_module(module_name)
            register_pickle_by_value(module)
    return dumps(obj)


def load_object(pickled: Union[bytes, str]) -> Any:
    """Deserialize an object from the payload."""
    if isinstance(pickled, str):
        return hex_to_python_object(pickled)
    else:
        return loads(pickled)


def python_object_to_name(obj: Any) -> Optional[str]:
    # Consider limiting the size of the name in future releases
    name = getattr(obj, "__name__", str(obj))

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
