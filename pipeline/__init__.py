import dill

import random
import string

import inspect

from pipeline.schemas import (
    PipelineVariableSchema,
    PipelineFunctionSchema,
    PipelineGraphNodeSchema,
    PipelineOutputVariableSchema,
    PipelineInputVariableSchema,
    PipelineGraph,
)


class Variable(object):
    schema: PipelineVariableSchema

    def __init__(self, **schema):
        if Pipeline._current_pipeline_defining:
            self.schema = PipelineVariableSchema(**schema)
            Pipeline.add_variable(self.schema)


class Pipeline(object):

    _current_pipeline: PipelineGraph = None
    _current_pipeline_defining = False

    def __init__(self):
        ...

    # __enter__ - called at the end of a "with" block.
    def __enter__(self):

        Pipeline._current_pipeline = PipelineGraph(
            inputs=[], outputs=[], variables=[], graph_nodes=[]
        )

        Pipeline._current_pipeline_defining = True
        return self

    # __exit__ - called at the end of a "with" block.
    def __exit__(self, type, value, traceback):
        Pipeline._current_pipeline_defining = False

    def output(self, *outputs):
        for _output in outputs:
            if not isinstance(_output, PipelineVariableSchema):
                raise Exception("Can only return PipelineVariables idiot.")

            variable_index = self._current_pipeline.variables.index(_output)
            if variable_index != -1:

                self._current_pipeline.variables[variable_index].is_output = True

                Pipeline._current_pipeline.outputs.append(
                    PipelineOutputVariableSchema(
                        variable=self._current_pipeline.variables[variable_index]
                    )
                )

    # Run the pipeline
    def run(self, *inputs):

        # Verify we have all of the inputs
        if len(inputs) != len(Pipeline._current_pipeline.inputs):
            raise Exception(
                "Mismatch of number of inputs, expecting %u got %s"
                % (len(Pipeline._current_pipeline.inputs), len(inputs))
            )
        running_variables = {}

        for i, input in enumerate(inputs):
            if not isinstance(
                input, Pipeline._current_pipeline.inputs[i].variable.variable_type
            ):
                raise Exception(
                    "Input type mismatch, expceted %s got %s"
                    % (
                        Pipeline._current_pipeline.inputs[i].variable.variable_type,
                        input.__class__,
                    )
                )
            running_variables[
                Pipeline._current_pipeline.inputs[i].variable.variable_name
            ] = input

        for node in Pipeline._current_pipeline.graph_nodes:
            node_inputs = node.inputs
            node_function = node.pipeline_function
            node_output = node.output

            function_inputs = []
            for _input in node_inputs:
                function_inputs.append(running_variables[_input.variable_name])

            output = node_function.function(*function_inputs)

            running_variables[node_output.variable_name] = output

        output_variables = []
        for output_variable in Pipeline._current_pipeline.outputs:
            output_variables.append(
                running_variables[output_variable.variable.variable_name]
            )

        return output_variables

    # def save(path):

    @staticmethod
    def add_variable(new_variable: PipelineVariableSchema):

        if Pipeline._current_pipeline_defining:

            Pipeline._current_pipeline.variables.append(new_variable)

            if new_variable.is_input:
                Pipeline._current_pipeline.inputs.append(
                    PipelineInputVariableSchema(variable=new_variable)
                )
            elif new_variable.is_output:
                Pipeline._current_pipeline.outputs.append(
                    PipelineOutputVariableSchema(variable=new_variable)
                )
        else:
            raise Exception("Cant add a variable when not defining a pipeline!")

    @staticmethod
    def get_pipeline():
        return Pipeline._current_pipeline


def pipeline_function(function):
    def execute_func(*args, **kwargs):

        if not Pipeline._current_pipeline_defining:
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
                elif isinstance(input_arg, PipelineVariableSchema):
                    if not input_arg in Pipeline._current_pipeline.variables:
                        raise Exception(
                            "Vairble not found, have you forgotten to define it as in input? "
                        )
                    processed_args.append(input_arg)
                elif hasattr(input_arg, "__pipeline_model__"):
                    print("ignore self")
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
