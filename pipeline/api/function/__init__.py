from pipeline.pipeline_schemas.function import FunctionCreate
from pipeline.api.call import post


def upload_function(function_create_schema: FunctionCreate):

    request_result = post("/v2/functions", function_create_schema.dict())

    return request_result
