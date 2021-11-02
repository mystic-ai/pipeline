import inspect
import requests
from dill import dumps

from npu2 import api, npu_print
from npu2 import pipeline
from npu2.pipeline import Pipeline
from npu2.api.function import upload_function
from npu2.api.pipeline import upload_pipeline


def upload(object):

    if hasattr(object, "__npu_func__"):
        function_name = object.__npu_func__.__name__
        function_bytes = dumps(object.__npu_func__)
        function_hex = function_bytes.hex()
        function_source = inspect.getsource(object.__npu_func__)

        return upload_function(function_name, function_hex, function_source)
    elif isinstance(object, Pipeline):
        uploaded_function_ids = []
        for function in object.pipeline_array:
            uploaded_function_ids.append(upload(function)["id"])

        pipeline_name = object.name
        return upload_pipeline(pipeline_name, functions=uploaded_function_ids)
    else:
        raise Exception("Not an npu2 object!")
