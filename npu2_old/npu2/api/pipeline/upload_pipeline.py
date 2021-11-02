from npu2.api.call import post


def upload_pipeline(pipeline_name="undefined_pipeline", functions=[]):
    pipeline_name = "undefined_pipeline" if pipeline_name == "" else pipeline_name
    print(pipeline_name)
    pipeline_dict = {
        "name": pipeline_name,
    }

    if functions != []:
        pipeline_dict["functions"] = functions
    return post("/pipeline", pipeline_dict)
