import inspect
import requests
from dill import dumps

from npu2 import api

def upload(object):

    if hasattr(object, "__npu_func__"):
        # Upload function
        headers = {
            "Authorization": "Bearer %s" % api.API_TOKEN
        }

        function_name = object.__npu_func__.__name__
        function_bytes = dumps(object.__npu_func__)
        function_hex = function_bytes.hex()
        function_source = inspect.getsource(object.__npu_func__)

        function_dict = {
            "name": function_name,
            "function_hex": function_hex,
            "function_source": function_source,
            "source_sample":"lol"
        }



        
        response = requests.post(api.API_ENDPOINT + "/function", headers=headers, json=function_dict)
        print(response.status_code)
        print(response.json()["source_sample"])