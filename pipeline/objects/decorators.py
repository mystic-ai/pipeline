from functools import partial, wraps

from pipeline.objects.function import Function
from pipeline.objects.graph_node import GraphNode
from pipeline.objects.model import Model
from pipeline.objects.pipeline import Pipeline
from pipeline.objects.variable import Variable


def pipeline_function(function=None, *, run_once=False, on_startup=False):
    """_summary_

    Args:
        function (callable, optional): _description_. This is the function to be
            wrapped, you do not pass this in manually it's automatically handled.

        run_once (bool, optional): _description_. Defaults to False. Setting to True
            will ensure that the decorated funciton is only called once
            in a pipeline run

        on_startup (bool, optional): _description_. Defaults to False. Setting to True
            will cause the wrapped function to be executed at the start of a pipeline
            run, regardless of when it's placed when defining the pipeline.

    """
    if function is None:
        return partial(pipeline_function, run_once=run_once, on_startup=on_startup)

    @wraps(function)
    def execute_func(*args, **kwargs):
        if not Pipeline._pipeline_context_active:
            return function(*args, **kwargs)
        else:
            function_ios = function.__annotations__
            if "return" not in function_ios:
                raise Exception(
                    "Must include an output type e.g. 'def my_func(...) -> int:'"
                )

            processed_args: Variable = []
            for input_arg in args:
                if isinstance(input_arg, Variable):
                    processed_args.append(input_arg)
                elif hasattr(input_arg, "__pipeline_model__"):
                    if function.__pipeline_function__.class_instance is None:
                        function.__pipeline_function__.class_instance = input_arg
                else:
                    raise Exception(
                        (
                            "You can't input random variables, "
                            "follow the way of the Pipeline. Got type %s"
                            % type(input_arg)
                        )
                    )

            node_output = Variable(type_class=function.__annotations__["return"])
            Pipeline.add_variable(node_output)
            Pipeline.add_function(function.__pipeline_function__)
            new_node = GraphNode(
                function=function.__pipeline_function__,
                inputs=processed_args,
                outputs=[node_output],
            )
            Pipeline.add_graph_node(new_node)

            return node_output

    execute_func.__function__ = function

    function.__run_once__ = run_once
    function.__has_run__ = False

    function.__on_startup__ = on_startup
    function.__pipeline_function__ = Function(function)

    return execute_func


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
            return self.__function_exe__(*args, **kwargs)

    def __function_exe__(self, *args, **kwargs):
        if not Pipeline._current_pipeline:
            return self.model_class(*args, **kwargs)
        else:
            created_model = self.model_class(*args, **kwargs)

            model_schema = Model(model=created_model)
            Pipeline._current_pipeline.models.append(model_schema)
            return created_model
