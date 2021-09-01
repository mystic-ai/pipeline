import inspect
import requests
from dill import dumps

from npu2 import api, npu_print
from npu2.api.function import create_function

def upload(object):

    if hasattr(object, "__npu_func__"):
        function_name = object.__npu_func__.__name__
        function_bytes = dumps(object.__npu_func__)
        function_hex = function_bytes.hex()
        function_source = inspect.getsource(object.__npu_func__)
        
        return create_function(function_name, function_hex, function_source)
    else:
        raise Exception("Not an npu2 object!")