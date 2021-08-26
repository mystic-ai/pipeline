import npu2

from npu2.object import Object
from npu2.pipeline import Pipeline, create_pipeline

class Model(Object):

    name: str
    predict_pipeline: Pipeline

    def __init__(self, name=""):
        self.name = name

        self.predict_pipeline = create_pipeline([self.pre_predict, self.predict, self.post_predict])
    
    @npu2.function()
    def pre_predict(*args, **kwargs):
        return args, kwargs
    
    @npu2.function()
    def predict(*args, **kwargs):
        return args, kwargs

    @npu2.function()
    def post_predict(*args, **kwargs):
        return args, kwargs