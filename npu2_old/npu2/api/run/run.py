from npu2.api.call import post

def run(run_target, data):
    # Run target can be a function or pipeline, just handle functions for the moment

    return post("/run/", {
        "function_id": run_target,
        "data": data,
    })