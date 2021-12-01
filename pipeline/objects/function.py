import inspect

from hashlib import sha256

from typing import Any, Callable, Dict, List, Optional
from pipeline import schemas

from pipeline.util import generate_id, hex_to_python_object

from pipeline.util import python_object_to_hex

from pipeline.schemas.file import FileCreate
from pipeline.schemas.function import FunctionGet, FunctionIOCreate, FunctionCreate


class Function:
    local_id: str
    remote_id: str

    name: str
    source: str
    hash: str

    typing_inputs: Dict[str, Any]
    typing_outputs: Dict[str, Any]

    function: Callable

    class_instance: Optional[Any]

    def __init__(
        self, function: Callable, *, remote_id: str = None, class_instance: Any = None
    ):
        self.name = function.__name__
        self.remote_id = remote_id
        self.class_instance = class_instance
        self.function = function

        self.source = inspect.getsource(function)
        self.hash = sha256(self.source.encode()).hexdigest()

        # TODO: Add verification that all inputs to function have a typing annotation, except for "self"
        if not "return" in function.__annotations__:
            raise Exception(
                "You must define an output type for a piepline function. e.g. def my_func(...) -> float:"
            )

        self.typing_outputs = {"return": function.__annotations__["return"]}

        self.typing_inputs = {
            function_i: function.__annotations__[function_i]
            for function_i in function.__annotations__
            if not function_i == "return"
        }

        self.local_id = generate_id(10)

    def to_create_schema(self) -> FunctionCreate:
        """
        inputs_schema = [
            FunctionIOCreate(
                name=variable_name,
                file=FileCreate(
                    name=variable_name,
                    file_bytes=python_object_to_hex(self.typing_inputs[variable_name]),
                ),
            )
            for variable_name in self.typing_inputs
        ]
        outputs_schema = [
            FunctionIOCreate(
                name=variable_name,
                file=FileCreate(
                    name=variable_name,
                    file_bytes=python_object_to_hex(self.typing_outputs[variable_name]),
                ),
            )
            for variable_name in self.typing_outputs
        ]
        """

        function_schema = FunctionCreate(
            local_id=self.local_id,
            name=self.name,
            function_source=self.source,
            hash=self.hash,
            # inputs=inputs_schema,
            # outputs=outputs_schema,
            file=FileCreate(name=self.name, file_bytes=python_object_to_hex(self)),
        )
        return function_schema

    @classmethod
    def from_schema(cls, schema: FunctionGet):
        # TODO: Add loading from files instead
        assert isinstance(schema, FunctionGet)
        function: Function = hex_to_python_object(schema.hex_file.data)
        # print(function.function(5.0))
        return function
