import os
import json
from dill import loads

def load_function(path: str):

    with open("%s/function_conf.json" % (path), "r") as conf_file:
        function_data = json.loads(conf_file.read())
        function_hex = function_data["data"]
        function_bytes = bytes.fromhex(function_hex)
        function = loads(function_bytes)
        return function

