from typing import Any

from pipeline.schemas.pipeline import PipelineVariableGet
from pipeline.util import generate_id, load_object


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
        if schema.pipeline_file_variable is not None:
            return PipelineFile.from_schema(schema)
        else:
            return cls(
                load_object(schema.type_file.data),
                is_input=schema.is_input,
                is_output=schema.is_output,
                name=schema.name,
                local_id=schema.local_id,
            )


class PipelineFile(Variable):

    path: str

    def __init__(
        self,
        *,
        path: str = None,
        name: str = None,
        remote_id: str = None,
        local_id: str = None,
    ) -> None:
        super().__init__(
            type_class=self.__class__,
            is_input=False,
            is_output=False,
            name=name,
            remote_id=remote_id,
            local_id=local_id,
        )
        self.path = path

    @classmethod
    def from_schema(cls, schema: PipelineVariableGet):
        return cls(
            path=schema.pipeline_file_variable.path,
            name=schema.name,
            local_id=schema.local_id,
        )
