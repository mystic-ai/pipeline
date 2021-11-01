import os
from pickle import load

from dill import loads, dumps

import random
import string

import inspect

from pipeline.schemas import (
    PipelineVariableSchema,
    PipelineFunctionSchema,
    PipelineGraphNodeSchema,
    PipelineGraph,
)

CACHE_DIR = os.getenv("PIPELINE_CACHE_DIR", "./cache")

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
elif not os.path.isdir(CACHE_DIR):
    raise Exception("Cache dir '%s' is not a valid dir." % CACHE_DIR)


class Variable(object):
    schema: PipelineVariableSchema

    def __init__(self, **schema):
        if Pipeline._current_pipeline_defining:
            self.schema = PipelineVariableSchema(**schema)
            Pipeline.add_variable(self.schema)


class Pipeline(object):
    defined_pipelines = {}

    _current_pipeline: PipelineGraph = None
    _current_pipeline_defining = False

    def __init__(self, pipeline_name):
        self.pipeline_name = pipeline_name

    # __enter__ - called at the end of a "with" block.
    def __enter__(self):

        Pipeline._current_pipeline = PipelineGraph(
            inputs=[],
            outputs=[],
            variables=[],
            graph_nodes=[],
            models=[],
            name=self.pipeline_name,
        )

        Pipeline._current_pipeline_defining = True
        return self

    # __exit__ - called at the end of a "with" block.
    def __exit__(self, type, value, traceback):
        Pipeline.defined_pipelines[self.pipeline_name] = self._current_pipeline.copy()

        Pipeline._current_pipeline_defining = False

    def output(self, *outputs):
        for _output in outputs:
            if not isinstance(_output, PipelineVariableSchema):
                raise Exception("Can only return PipelineVariables idiot.")

            variable_index = self._current_pipeline.variables.index(_output)
            if variable_index != -1:

                self._current_pipeline.variables[variable_index].is_output = True

                """Pipeline._current_pipeline.outputs.append(
                    PipelineOutputVariableSchema(
                        variable=self._current_pipeline.variables[variable_index]
                    )
                )"""

    def save(self, dir, **kwargs):
        return self._current_pipeline.save(dir, **kwargs)

    @staticmethod
    def load(path):
        # with open(os.path.join(path, "config.json"), "r") as config_file:
        Pipeline._current_pipeline = PipelineGraph.parse_file(
            os.path.join(path, "config.json")
        )

        # Load functions in graph nodes
        for node in Pipeline._current_pipeline.graph_nodes:
            function_file_name = node.pipeline_function.function_file_name
            with open(os.path.join(path, function_file_name), "rb") as function_file:
                function_data = loads(function_file.read())
                node.pipeline_function.function = function_data
            if node.pipeline_function.bound_class_file_name != None:
                with open(
                    os.path.join(path, node.pipeline_function.bound_class_file_name),
                    "rb",
                ) as cls_file:
                    cls_data = loads(cls_file.read())
                    node.pipeline_function.bound_class = cls_data

        # Load models
        for model in Pipeline._current_pipeline.models:
            with open(
                os.path.join(path, model.model_file_name),
                "rb",
            ) as model_file:
                model_data = loads(model_file.read())
                model.model = model_data
        # Load variables
        for variable in Pipeline._current_pipeline.variables:
            with open(
                os.path.join(path, variable.variable_type_file_path),
                "rb",
            ) as variable_type_file:
                variable_type_data = loads(variable_type_file.read())
                variable.variable_type = variable_type_data

        return Pipeline._current_pipeline

    # def save(path):

    @staticmethod
    def add_variable(new_variable: PipelineVariableSchema):

        if Pipeline._current_pipeline_defining:

            Pipeline._current_pipeline.variables.append(new_variable)

            """if new_variable.is_input:
                Pipeline._current_pipeline.inputs.append(
                    PipelineInputVariableSchema(variable=new_variable)
                )
            elif new_variable.is_output:
                Pipeline._current_pipeline.outputs.append(
                    PipelineOutputVariableSchema(variable=new_variable)
                )"""
        else:
            raise Exception("Cant add a variable when not defining a pipeline!")

    @staticmethod
    def get_pipeline(pipeline_name) -> PipelineGraph:
        return Pipeline.defined_pipelines[pipeline_name]


def pipeline_function(function):
    def execute_func(*args, **kwargs):

        if not Pipeline._current_pipeline_defining:
            print("Call")
            return function(*args, **kwargs)
        else:

            function_ios = function.__annotations__
            if not "return" in function_ios:
                raise Exception(
                    "Must include an output type e.g. 'def my_func(...) -> int:'"
                )

            processed_args = []

            for input_arg in args:
                if isinstance(input_arg, Variable):
                    if not input_arg.schema in Pipeline._current_pipeline.variables:
                        raise Exception(
                            "Vairble not found, have you forgotten to define it as in input? "
                        )
                    processed_args.append(input_arg.schema)
                elif hasattr(input_arg, "__pipeline_model__"):
                    if function.__pipeline_function__.bound_class == None:
                        function.__pipeline_function__.bound_class = input_arg
                elif isinstance(input_arg, PipelineVariableSchema):
                    print(input_arg)
                    if not input_arg in Pipeline._current_pipeline.variables:
                        raise Exception(
                            "Vairble not found, have you forgotten to define it as in input? "
                        )
                    processed_args.append(input_arg)
                else:
                    raise Exception(
                        "You can't input random variables, follow the way of the Pipeline. Got type"
                    )

            node_output = PipelineVariableSchema(
                variable_type=function.__annotations__["return"]
            )
            Pipeline.add_variable(node_output)

            # Everytime this function is called we have to add a new node in the graph
            graph_node = PipelineGraphNodeSchema(
                pipeline_function=function.__pipeline_function__,
                inputs=processed_args,
                output=node_output,
            )
            Pipeline._current_pipeline.graph_nodes.append(graph_node)

            return graph_node.output

    function_inputs = {
        function_i: function.__annotations__[function_i]
        for function_i in function.__annotations__
        if not function_i == "return"
    }

    function.__pipeline_function__ = PipelineFunctionSchema(
        inputs=function_inputs,
        name=function.__name__,
        function=function,
    )

    return execute_func
