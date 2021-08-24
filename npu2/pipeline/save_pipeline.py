import os
import json

from npu2.pipeline import __pipline_to_dict__
from npu2.pipeline import Pipeline

def save_pipeline(pipeline: Pipeline, path: str):
    pipeline_dict = __pipline_to_dict__(pipeline)

    with open("%s/model.json" % (path), "w") as conf_file:
        conf_file.write(json.dumps({"name":pipeline.name, "pipeline_array_length":len(pipeline.pipeline_array)}))
    
    for index, pipeline_func in enumerate(pipeline_dict["pipeline_array"]):
        if not os.path.exists("%s/" % (path)):
            os.makedirs("%s/" % (path))

        with open("%s/%u" % (path, index), "wb") as pipline_func_file:
            pipline_func_file.write(pipeline_func)

