import string
import random
import inspect

from dill import dumps

from hashlib import sha256

from typing import Optional, Callable, Any, Union

from pipeline.pipeline_schemas.function import FunctionCreate


class PipelineFunction(object):
    local_id: str = None
    api_id: Optional[str] = None

    name: str = None
    hash: Optional[str] = None
    inputs: dict = {}
    output: dict = {}

    function_hex: str = None
    function_source: str = None

    function: Optional[Callable] = None
    function_file_name: Optional[str] = None

    bound_class: Optional[Any] = None
    bound_class_file_name: Optional[str] = None

    def __init__(self, function: Callable):
        self.local_id = "".join(
            random.choice(string.ascii_lowercase) for i in range(20)
        )
        self.name = function.__name__

        self.inputs = {
            function_i: function.__annotations__[function_i]
            for function_i in function.__annotations__
            if not function_i == "return"
        }

        self.output = {"return": function.__annotations__["return"].__name__}

        self.function = function

        self.function_source = inspect.getsource(function)
        self.function_hex = dumps(function).hex()
        self.hash = sha256(self.function_source.encode()).hexdigest()

    def dict(self, *args, **kwargs):
        return dict(
            local_id=self.local_id,
            remote_id=self.remote_id,
            inputs=self.inputs,
            name=self.name,
            hash=self.hash,
            function_file_name=self.function_file_name,
            bound_class_file_name=self.bound_class_file_name,
        )

    def json(self, *args, **kwargs):
        return self.dict()

    @property
    def _api_create_schema(self) -> FunctionCreate:
        create_schema = FunctionCreate(
            name=self.name,
            hash=self.hash,
            function_hex=self.function_hex,
            function_source=self.function_source,
            inputs={_input: self.inputs[_input].__name__ for _input in self.inputs},
            output=self.output,
        )
        return create_schema

    def _set_bound_class(self, bound_class) -> None:
        self.bound_class = bound_class
