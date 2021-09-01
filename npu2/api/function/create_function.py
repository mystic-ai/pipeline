from npu2.api.call import post

def create_function(function_name, function_hex, function_source):
    function_dict = {
        "name": function_name,
        "function_hex": function_hex,
        "function_source": function_source,
        "source_sample":function_source[:min(200, len(function_source))]
    }
    return post("/function", function_dict)
