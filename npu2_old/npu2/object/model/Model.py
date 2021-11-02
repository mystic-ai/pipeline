from types import MethodType
import types

import npu2

from npu2.object import Object
from npu2.pipeline import Pipeline, create_pipeline

class Model(Object):

    name: str
    predict_pipeline: Pipeline
    train_pipeline: Pipeline

    def __init__(self, name=""):
        self.name = name
        self.update_pipelines()
    
    def __bound_check__(self, func):        
        if isinstance(func, types.FunctionType):
            return MethodType(func, self)
        else:
            return func

    def update_pipelines(self):
        self.pre_predict = self.__bound_check__(self.pre_predict)
        self.predict = self.__bound_check__(self.predict)
        self.post_predict = self.__bound_check__(self.post_predict)
        

        self.predict_pipeline = create_pipeline([self.pre_predict, self.predict, self.post_predict])

    @npu2.function()
    def pre_predict(self, *args, **kwargs):
        return args, kwargs
    
    @npu2.function()
    def predict(self, *args, **kwargs):
        return args, kwargs

    @npu2.function()
    def post_predict(self, *args, **kwargs):
        return args, kwargs