import inspect
import uuid
from hashlib import sha256
from typing import Any

from pipeline.util import generate_id


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
        try:
            self.source = inspect.getsource(model.__class__)
        except OSError:
            self.source = str(uuid.uuid4())

        self.hash = sha256(self.source.encode()).hexdigest()
        self.local_id = generate_id(10) if local_id is None else local_id
        if not hasattr(self.model, "local_id"):
            setattr(self.model, "local_id", self.local_id)
