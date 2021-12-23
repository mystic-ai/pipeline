from typing import Any


from pipeline.util import generate_id


class Model:
    local_id: str
    remote_id: str

    name: str

    model: Any

    def __init__(self, model: Any, *, name: str = "", local_id: str = None):

        self.name = name
        self.model = model

        self.local_id = generate_id(10) if local_id == None else local_id
