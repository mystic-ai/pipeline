import inspect
from hashlib import sha256
from typing import Any

from pipeline.util import generate_id, hex_to_python_object

from pipeline.schemas.model import ModelGet


class Model:
    local_id: str
    remote_id: str

    name: str
    source: str
    hash: str

    model: Any

    def __init__(self, model: Any, *, name: str = "", local_id: str = None):

        self.name = name
        self.model = model
        self.source = inspect.getsource(model.__class__)
        self.hash = sha256(self.source.encode()).hexdigest()
        self.local_id = generate_id(10) if local_id is None else local_id
        setattr(model, "__local_id__", self.local_id)

    @classmethod
    def from_schema(cls, schema: ModelGet):
        return cls(
            hex_to_python_object(schema.hex_file.data),
            name=schema.name,
            local_id=schema.id,
        )
