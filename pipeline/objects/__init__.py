from pipeline.objects.variable import Variable
from pipeline.objects.function import Function
from pipeline.objects.graph_node import GraphNode
from pipeline.objects.graph import Graph
from pipeline.objects.pipeline import Pipeline


def pipeline_function(function):
    def execute_func(*args, **kwargs):

        if not Pipeline._pipeline_context_active:
            return function(*args, **kwargs)
        else:
            function_ios = function.__annotations__
            if not "return" in function_ios:
                raise Exception(
                    "Must include an output type e.g. 'def my_func(...) -> int:'"
                )

            processed_args: Variable = []

            for input_arg in args:
                if isinstance(input_arg, Variable):
                    processed_args.append(input_arg)
                else:
                    raise Exception(
                        "You can't input random variables, follow the way of the Pipeline. Got type"
                    )

            node_output = Variable(type_class=function.__annotations__["return"])
            # Pipeline.add_variable(node_output)
            Pipeline.add_function(function.__pipeline_function__)
            new_node = GraphNode(
                function=function.__pipeline_function__,
                inputs=processed_args,
                outputs=[node_output],
            )
            Pipeline.add_graph_node(new_node)

            return node_output

    execute_func.__function__ = function
    function.__pipeline_function__ = Function(function)
    return execute_func
