import os
import json
from dill import dumps
import inspect

def save_function(function, path: str):
    if not os.path.exists("%s/" % (path)):
            os.makedirs("%s/" % (path))

    if not hasattr(function, "__npu_func__"):
        #print(function)
        raise Exception("Not an npu function")

    with open("%s/function_conf.json" % (path), "w") as conf_file:
        function_bytes = dumps(function)
        function_hex = function_bytes.hex()
        function_source = inspect.getsource(function.__npu_func__)

        function_dict = {"name": function.__npu_func__.__name__, "source": function_source, "data":function_hex}

        conf_file.write(json.dumps(function_dict))

