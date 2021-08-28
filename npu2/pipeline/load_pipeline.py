import os
import json

from npu2.pipeline import __pipline_from_dict__
from npu2.pipeline import Pipeline

from npu2.function import load_function

def load_pipeline(path: str):
    pipeline_dict = {}

    with open("%s/pipeline_conf.json" % (path), "r") as conf_file:
        pipeline_dict = json.loads(conf_file.read())

    pipeline_dict["pipeline_array"] = []

    for index in range(pipeline_dict["pipeline_array_length"]):
        pipeline_func = load_function("%s/%u" % (path, index))
        pipeline_dict["pipeline_array"].append(pipeline_func)

    pipeline = __pipline_from_dict__(pipeline_dict)
    return pipeline

