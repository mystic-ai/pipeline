
from dill import loads, dumps
from npu2.object.model.Model import Model
'''
def __model_to_dict__(model: Model):
    model_dict = {}

    model_dict["pipelines"] = [dumps(pipeline_conf) for pipeline_conf in pipeline.pipeline_array]
    model_dict["name"] = model.name

    return model_dict

def __pipline_from_dict__(pipeline_dict: dict):

    pipeline_array = [loads(pipeline_func) for pipeline_func in pipeline_dict["pipeline_array"]]
    pipeline_name = pipeline_dict["name"]
    new_pipeline = Pipeline(name=pipeline_name, pipeline_array=pipeline_array)
    return new_pipeline
'''