import string
import random
from pydantic import BaseModel
from typing import List, Optional, Callable

is_in_with = False


class PipelineVariable(BaseModel):
    variable_type: type
    variable_name: str
    is_input: bool = False
    is_output: bool = False

    def __init__(self, **kwargs):
        if not "variable_name" in kwargs:
            kwargs["variable_name"] = "".join(
                random.choice(string.ascii_lowercase) for i in range(10)
            )
        super().__init__(**kwargs)

        global is_in_with
        if is_in_with:
            global current_pipeline
            current_pipeline.variables.append(self)

            if self.is_input:
                current_pipeline.inputs.append(PipelineInputVariable(variable=self))
            elif self.is_output:
                current_pipeline.outputs.append(PipelineOutputVariable(variable=self))


class PipelineInputVariable(BaseModel):
    variable: PipelineVariable


class PipelineOutputVariable(BaseModel):
    variable: PipelineVariable


class PipelineFunction(BaseModel):
    inputs: dict
    name: str
    hash: Optional[str]
    function: Callable


class PipelineGraphNode(BaseModel):
    pipeline_function: PipelineFunction
    inputs: List[PipelineVariable]
    output: PipelineVariable


class PipelineGraph(BaseModel):
    inputs: List[PipelineInputVariable]
    variables: List[PipelineVariable]
    outputs: List[PipelineOutputVariable]
    graph_nodes: List[PipelineGraphNode]


class Pipeline(object):
    def __init__(self):
        ...

    def __enter__(self):
        global is_in_with

        global current_pipeline
        global current_pipeline_str
        current_pipeline = PipelineGraph(
            inputs=[], outputs=[], variables=[], graph_nodes=[]
        )
        current_pipeline_str = "".join(
            random.choice(string.ascii_lowercase) for i in range(10)
        )

        is_in_with = True
        return self

    def output(self, *outputs):
        for _output in outputs:
            if not isinstance(_output, PipelineVariable):
                raise Exception("Can only return PipelineVariables idiot.")
            global current_pipeline

            _output.is_output = True

            current_pipeline.outputs.append(PipelineOutputVariable(variable=_output))

    def run(self, *inputs):
        global current_pipeline
        # Verify we have all of the
        if len(inputs) != len(current_pipeline.inputs):
            raise Exception(
                "Mismatch of number of inputs, expecting %u got %s"
                % (len(current_pipeline.inputs), len(inputs))
            )
        running_variables = {}

        for i, input in enumerate(inputs):
            if not isinstance(input, current_pipeline.inputs[i].variable.variable_type):
                raise Exception(
                    "Input type mismatch, expceted %s got %s"
                    % (
                        current_pipeline.inputs[i].variable.variable_type,
                        input.__class__,
                    )
                )
            running_variables[current_pipeline.inputs[i].variable.variable_name] = input

        for node in current_pipeline.graph_nodes:
            node_inputs = node.inputs
            node_function = node.pipeline_function
            node_output = node.output

            function_inputs = []
            for _input in node_inputs:
                function_inputs.append(running_variables[_input.variable_name])

            output = node_function.function(*function_inputs)

            running_variables[node_output.variable_name] = output

        output_variables = []
        for output_variable in current_pipeline.outputs:
            output_variables.append(
                running_variables[output_variable.variable.variable_name]
            )

        return output_variables

    def __exit__(self, type, value, traceback):
        global is_in_with
        is_in_with = False


def pipeline_function(function):
    def execute_func(*args, **kwargs):
        global is_in_with
        if not is_in_with:
            return function(*args, **kwargs)
        else:
            global current_pipeline

            function_ios = function.__annotations__
            if not "return" in function_ios:
                raise Exception(
                    "Must include an output type e.g. 'def my_func(...) -> int:'"
                )

            for input_arg in args:
                if not isinstance(input_arg, PipelineVariable):
                    raise Exception(
                        "You can't input random variables, follow the way of the Pipeline."
                    )
                if not input_arg in current_pipeline.variables:
                    raise Exception(
                        "Vairble not found, have you forgotten to define it as in input? "
                        % current_pipeline
                    )

            # Everytime this function is called we have to add a new node in the graph
            graph_node = PipelineGraphNode(
                pipeline_function=function.__pipeline_function__,
                inputs=args,
                output=PipelineVariable(
                    variable_type=function.__annotations__["return"]
                ),
            )
            current_pipeline.graph_nodes.append(graph_node)

            for input_arg in args:
                if not isinstance(input_arg, PipelineVariable):
                    ...

            return graph_node.output

    function_inputs = {
        function_i: function.__annotations__[function_i]
        for function_i in function.__annotations__
        if not function_i == "return"
    }

    function.__pipeline_function__ = PipelineFunction(
        inputs=function_inputs,
        name=function.__name__,
        function=function,
    )
    return execute_func


# Testing code
current_pipeline_str: str = None
current_pipeline: PipelineGraph = None


@pipeline_function
def square(a: float) -> float:
    return a ** 2


@pipeline_function
def add(a: float, b: float) -> float:
    return a + b


@pipeline_function
def minus(a: float, b: float) -> float:
    return a - b


@pipeline_function
def multiply(a: float, b: float) -> float:
    return a * b


with Pipeline() as pipeline:
    flt_1 = PipelineVariable(variable_type=float, is_input=True)
    flt_2 = PipelineVariable(variable_type=float, is_input=True)
    sq_1 = square(flt_1)
    res_1 = multiply(flt_2, sq_1)

    pipeline.output(res_1, sq_1)

print(pipeline.run(5.0, 6.0))
