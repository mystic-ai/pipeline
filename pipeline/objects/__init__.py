from pipeline.objects.function import Function
from pipeline.objects.graph_node import GraphNode
from pipeline.objects.model import Model
from pipeline.objects.paiplain import Paiplain
from pipeline.objects.pipeline import Pipeline
from pipeline.objects.variable import Variable

__all__ = ["Paiplain"]


class pipeline_model(object):
    def __init__(
        self, model_class=None, *, file_or_dir: str = None, compress_tar=False
    ):
        if model_class is not None:
            model_class.__pipeline_model__ = True

        self.compress_tar = compress_tar
        self.model_class = model_class
        self.file_or_dir = file_or_dir

    def __call__(self, *args, **kwargs):

        if len(args) + len(kwargs) == 1:
            self.model_class = args[0]
            self.model_class.__pipeline_model__ = True
            return self.__function_exe__
        else:
            print(len(args) + len(kwargs))
            return self.__function_exe__(*args, **kwargs)

    def __function_exe__(self, *args, **kwargs):
        if not Pipeline._current_pipeline:
            return self.model_class(*args, **kwargs)
        else:
            created_model = self.model_class(*args, **kwargs)
            model_schema = Model(model=created_model)
            Pipeline._current_pipeline.models.append(model_schema)

            # model_functions = [
            #     model_attr
            #     for model_attr in dir(created_model)
            #     if callable(getattr(created_model, model_attr))
            #     and model_attr[:2] != "__"
            # ]
            return created_model
