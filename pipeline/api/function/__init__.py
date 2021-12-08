import io

from pydantic.schema import field_schema
from pipeline.util import python_object_to_name, python_object_to_hex

from pipeline.objects.function import Function
from pipeline.schemas.function import FunctionCreate, FunctionGet

from pipeline.api.call import post
from pipeline.api.file import upload_file, upload_python_object_to_file


def upload_function(function: Function) -> FunctionGet:

    inputs = [
        dict(name=name, type_name=python_object_to_name(type))
        for name, type in function.typing_inputs.items()
    ]
    output = [
        dict(name=name, type_name=python_object_to_name(type))
        for name, type in function.typing_outputs.items()
    ]

    file_schema = upload_python_object_to_file(function, "/lol")

    function_create_schema = FunctionCreate(
        local_id=function.local_id,
        name=function.name,
        function_source=function.source,
        hash=function.hash,
        inputs=inputs,
        output=output,
        file_id=file_schema.id,
    )
    print(function_create_schema.json())
    request_result = post("/v2/functions/", function_create_schema.dict())

    return FunctionGet.parse_obj(request_result)
