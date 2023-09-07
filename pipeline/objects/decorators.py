import typing
from functools import partial, wraps

from pipeline.objects.function import Function
from pipeline.objects.graph import GraphNode, Variable
from pipeline.objects.model import Model
from pipeline.objects.pipeline import Pipeline


def pipe(function=None, *, run_once=False, on_startup=False):
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
        return partial(pipe, run_once=run_once, on_startup=on_startup)

    @wraps(function)
    def execute_func(*args, **kwargs):
        if not Pipeline._pipeline_context_active:
            return function(*args, **kwargs)
        else:
            if "return" not in function.__annotations__:
                raise Exception(
                    (
                        "You must define an output type for a pipeline function. "
                        "e.g. def my_func(...) -> float:"
                    )
                )

            processed_args: Variable = []
            for input_arg in args:
                if isinstance(input_arg, Variable):
                    processed_args.append(input_arg)
                elif hasattr(input_arg, "__entity__"):
                    if function.__pipe__.class_instance is None:
                        function.__pipe__.class_instance = input_arg
                elif isinstance(input_arg, tuple) and all(
                    isinstance(var, Variable) for var in input_arg
                ):
                    raise Exception(
                        "Must seperate outputs from functions with Tuple outputs:"
                        "`var1, var2, ..., varN = func(...)`",
                    )

                else:
                    print(hasattr(function, "__self__"))
                    print(f"Function: {function}, Input: {input_arg}")
                    # breakpoint()

                    if function in input_arg.__dict__.values():
                        print("Function found in class")
                    else:
                        print(input_arg.__dict__.values())

                    raise Exception(
                        (
                            f"Can only input pipeline variables to a function "
                            f"when defining a graph, got: {type(input_arg)}"
                        )
                    )

            node_outputs: typing.List[Variable] = []

            function_output = function.__annotations__["return"]
            if getattr(function_output, "__origin__", None) == tuple:
                context_manager_variables = node_outputs = tuple(
                    Variable(type_class=output_variable, is_input=False)
                    for output_variable in function_output.__args__
                )

            else:
                context_manager_variables = Variable(
                    type_class=function.__annotations__["return"],
                    is_input=False,
                )
                node_outputs = [context_manager_variables]

            # Pipeline.add_variables(*node_outputs)
            Pipeline.add_function(function.__pipe__)

            new_node = GraphNode(
                function=function.__pipe__,
                inputs=processed_args,
                outputs=node_outputs,
            )

            Pipeline.add_graph_node(new_node)

            return context_manager_variables

    execute_func.__function__ = function

    function.__run_once__ = run_once
    function.__has_run__ = False

    function.__on_startup__ = on_startup
    function.__pipe__ = Function(function)

    return execute_func


class entity(object):
    def __init__(
        self,
        model_class=None,
    ):
        if model_class is not None:
            model_class.__entity__ = True

        self.model_class = model_class

    def __call__(self, *args, **kwargs):
        if len(args) + len(kwargs) == 1:
            self.model_class = args[0]
            self.model_class.__entity__ = True
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
