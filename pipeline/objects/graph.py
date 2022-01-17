from typing import List

from dill import dumps, loads

from pipeline.objects.function import Function
from pipeline.objects.graph_node import GraphNode
from pipeline.objects.model import Model
from pipeline.objects.variable import Variable
from pipeline.schemas.pipeline import PipelineGet
from pipeline.util import generate_id


class Graph:
    local_id: str
    remote_id: str

    name: str

    functions: List[Function]
    variables: List[Variable]

    outputs: List[Variable]

    nodes: List[GraphNode]

    models: List[Model]
    # TODO: Add generic objects (e.g. Model) to be included in the graph

    def __init__(
        self,
        *,
        name: str = "",
        variables: List[Variable] = None,
        functions: List[Function] = None,
        outputs: List[Variable] = None,
        nodes: List[GraphNode] = None,
        models: List[Model] = None,
    ):
        self.name = name
        self.local_id = generate_id(10)

        self.variables = variables if variables is not None else []
        self.functions = functions if functions is not None else []
        self.outputs = outputs if outputs is not None else []
        self.nodes = nodes if nodes is not None else []
        self.models = models if models is not None else []

    def run(self, *inputs):
        input_variables: List[Variable] = [
            var for var in self.variables if var.is_input
        ]

        # TODO: Add generic object loading

        if len(inputs) != len(input_variables):
            raise Exception(
                "Mismatch of number of inputs, expecting %u got %s"
                % (len(input_variables), len(inputs))
            )

        for model in self.models:
            if hasattr(model.model, "load"):
                model.model.load(None)

        running_variables = {}
        for i, input in enumerate(inputs):
            if not isinstance(input, input_variables[i].type_class):
                raise Exception(
                    "Input type mismatch, expceted %s got %s"
                    % (
                        input_variables[i].type_class,
                        input.__class__,
                    )
                )
            running_variables[input_variables[i].local_id] = input

        for node in self.nodes:

            node_inputs: List[Variable] = []
            node_outputs: List[Variable] = []
            node_function: Function = None

            for _node_input in node.inputs:
                for variable in self.variables:
                    if variable.local_id == _node_input.local_id:
                        node_inputs.append(variable)
                        break
            for _node_output in node.outputs:
                for variable in self.variables:
                    if variable.local_id == _node_output.local_id:
                        node_outputs.append(variable)
                        break

            for function in self.functions:
                if function.local_id == node.function.local_id:
                    node_function = function
                    break

            function_inputs = []
            for _input in node_inputs:
                function_inputs.append(running_variables[_input.local_id])

            if (
                hasattr(node.function, "class_instance")
                and node_function.class_instance is not None
            ):
                output = node_function.function(
                    node_function.class_instance, *function_inputs
                )
            else:
                output = node_function.function(*function_inputs)

            running_variables[node_outputs[0].local_id] = output

        return_variables = []

        for output_variable in self.outputs:
            return_variables.append(running_variables[output_variable.local_id])

        return return_variables

    """
    def to_create_schema(self) -> PipelineCreate:
        variables = [_var.to_create_schema() for _var in self.variables]
        functions = [_func.to_create_schema() for _func in self.functions]

        graph_nodes = [_node.to_create_schema() for _node in self.nodes]

        create_schema = PipelineCreate(
            name=self.name,
            variables=variables,
            functions=functions,
            graph_nodes=graph_nodes,
            outputs=[_var.local_id for _var in self.outputs],
        )

        return create_schema
    """

    @classmethod
    def from_schema(cls, schema: PipelineGet):
        variables = [Variable.from_schema(_var) for _var in schema.variables]
        functions = [Function.from_schema(_func) for _func in schema.functions]
        outputs = []
        for _output in schema.outputs:
            for _var in variables:
                if _var.local_id == _output:
                    outputs.append(_var)

        nodes = []

        for _node in schema.graph_nodes:
            node_inputs = []
            for node_str in _node.inputs:
                for _var in variables:
                    if _var.local_id == node_str:
                        node_inputs.append(_var)
                        break

            node_outputs = []
            for node_str in _node.outputs:
                for _var in variables:
                    if _var.local_id == node_str:
                        node_outputs.append(_var)
                        break

            function = None

            for _func in functions:
                if _func.local_id == _node.function:
                    function = _func
                    break

            if function is None:
                raise Exception("Function not found:%s" % _node.function)

            nodes.append(
                GraphNode(
                    function=function,
                    inputs=node_inputs,
                    outputs=node_outputs,
                    local_id=_node.local_id,
                )
            )

        remade_graph = cls(
            name=schema.name,
            variables=variables,
            functions=functions,
            outputs=outputs,
            nodes=nodes,
        )

        return remade_graph

    def save(self, save_path):
        with open(save_path, "wb") as save_file:
            save_file.write(dumps(self))

    @classmethod
    def load(cls, load_path):
        with open(load_path, "rb") as load_file:
            return loads(load_file.read())
