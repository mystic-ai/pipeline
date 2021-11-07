from pipeline.api.call import post


def upload_function(function_name, function_hex, function_source):
    function_dict = {
        "name": function_name,
        "function_hex": function_hex,
        "function_source": function_source,
        "source_sample": function_source[: min(200, len(function_source))],
    }
    request_result = post("/v2/functions", function_dict)

    return request_result
