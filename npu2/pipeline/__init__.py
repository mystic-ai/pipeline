from os import name
from dill import dumps, loads
from npu2.pipeline.Pipeline import Pipeline

def __pipline_to_dict__(pipeline: Pipeline):
    pipeline_dict = {}

    pipeline_dict["pipeline_array"] = [(pipeline_conf) for pipeline_conf in pipeline.pipeline_array]
    pipeline_dict["name"] = pipeline.name

    return pipeline_dict

def __pipline_from_dict__(pipeline_dict: dict):

    pipeline_array = [(pipeline_func) for pipeline_func in pipeline_dict["pipeline_array"]]
    pipeline_name = pipeline_dict["name"]
    new_pipeline = Pipeline(name=pipeline_name, pipeline_array=pipeline_array)
    return new_pipeline

from npu2.pipeline.create_pipeline import create_pipeline
from npu2.pipeline.save_pipeline import save_pipeline
from npu2.pipeline.load_pipeline import load_pipeline
from npu2.pipeline.validate_pipeline import validate_pipeline
