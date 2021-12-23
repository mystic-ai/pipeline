from typing import Any


from pipeline.util import (
    generate_id,
    python_object_to_hex,
    python_object_to_name,
    hex_to_python_object,
)

from pipeline.schemas.file import FileCreate
from pipeline.schemas.pipeline import PipelineVariableGet


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
        local_id: str = None
    ):
        self.remote_id = remote_id
        self.name = name
        self.type_class = type_class
        self.is_input = is_input
        self.is_output = is_output

        self.local_id = generate_id(10) if not local_id else local_id

        if Pipeline._pipeline_context_active:
            Pipeline.add_variable(self)

    @classmethod
    def from_schema(cls, schema: PipelineVariableGet):
        return cls(
            hex_to_python_object(schema.type_file.data),
            is_input=schema.is_input,
            is_output=schema.is_output,
            name=schema.name,
            remote_id=schema.remote_id,
            local_id=schema.local_id,
        )


from pipeline.objects.pipeline import Pipeline
