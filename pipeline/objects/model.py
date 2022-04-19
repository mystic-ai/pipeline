import inspect
from hashlib import sha256
from typing import Any

from pipeline.schemas.model import ModelGet
from pipeline.util import generate_id, hex_to_python_object


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
        if not hasattr(self.model, "local_id"):
            setattr(self.model, "local_id", self.local_id)

    @classmethod
    def from_schema(cls, schema: ModelGet):
        pickled_data = hex_to_python_object(schema.hex_file.data)
        if isinstance(pickled_data, Model):
            pickled_data.local_id = schema.id
            return pickled_data
        return cls(
            pickled_data,
            name=schema.name,
            local_id=schema.id,
        )
