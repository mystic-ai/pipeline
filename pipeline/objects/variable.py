from typing import Any

from pipeline.schemas.pipeline import PipelineVariableGet
from pipeline.util import generate_id, hex_to_python_object


class Variable:

    local_id: str
    remote_id: str

    name: str

    type_class: Any

    is_input: bool
    is_output: bool

    def __init__(
        self,
        type_class: Any,
        *,
        is_input: bool = False,
        is_output: bool = False,
        name: str = None,
        remote_id: str = None,
        local_id: str = None,
    ):
        self.remote_id = remote_id
        self.name = name
        self.type_class = type_class
        self.is_input = is_input
        self.is_output = is_output

        self.local_id = generate_id(10) if not local_id else local_id

    @classmethod
    def from_schema(cls, schema: PipelineVariableGet):
        return cls(
            hex_to_python_object(schema.type_file.data),
            is_input=schema.is_input,
            is_output=schema.is_output,
            name=schema.name,
            local_id=schema.local_id,
        )
