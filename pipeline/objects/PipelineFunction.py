import string
import random
import inspect

from dill import dumps

from hashlib import sha256

from typing import Optional, Callable, Any, Union

from pipeline.pipeline_schemas.function import FunctionCreate


def PipelineFunction(object):
    local_id: str
    api_id: Optional[str]

    name: str
    hash: Optional[str]
    inputs: dict
    output: dict

    function_hex: str
    function_source: str

    function: Optional[Callable]
    function_file_name: Optional[str]

    bound_class: Optional[Any]
    bound_class_file_name: Optional[str]

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

        self.output = function.__annotations__["return"]

        self.function = function

        self.function_source = inspect.getsource(function)
        self.function_hex = dumps(function).hex()
        self.hash = sha256(self.function_source).hexdigest()

    @property
    def _api_create_schema(self) -> FunctionCreate:
        create_schema = FunctionCreate(
            name=self.name,
            hash=self.hash,
            function_hex=self.function_hex,
            function_source=self.function_source,
            inputs=self.inputs,
            output=self.output,
        )
        return create_schema
