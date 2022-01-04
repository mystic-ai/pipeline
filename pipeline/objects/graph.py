from __future__ import annotations

from typing import Any, List, Tuple

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

    # functions: List[Function]
    # FIXME
    # for some reason a graph is accesing all variables added to all graphs
    # with belongs_to value we can ignore the incorrect variables, but
    # not having them spill over at all would be much better
    # variables: List[Variable]
    # outputs: List[Variable]

    nodes: List[GraphNode]

    models: List[Model]
    # TODO: Add generic objects (e.g. Model) to be included in the graph

    def __init__(
        self,
        *,
        name: str = "",
        # variables: List[Variable] = [],
        # functions: List[Function] = [],
        # outputs: List[Variable] = [],
        nodes: List[GraphNode] = [],
        models: List[Model] = [],
    ):
        self.name = name
        self.local_id = generate_id(10)

        # self.variables = variables
        # self.functions = functions
        # self.outputs = outputs
        self.nodes = nodes
        self.models = models

    def add_node(self, node: GraphNode) -> None:
        self.nodes.append(node)

    # def add_function(self, function: Function) -> None:
    #     self.functions.append(function)

    # def add_variable(self, variable: Variable) -> Variable:
    #     if variable not in self.variables:
    #         self.variables.append(variable)
    #     return variable

    # def run(self, *inputs) -> List[Any]:
    #     input_variables: List[Variable] = [
    #         var
    #         for var in self.variables
    #         if var.is_input and var.belongs_to == self.name
    #     ]

    #     # TODO: Add generic object loading
    #     if len(inputs) != len(input_variables):
    #         raise Exception(
    #             "Mismatch of number of inputs, expecting %u got %s"
    #             % (len(input_variables), len(inputs))
    #         )

    #     for model in self.models:
    #         if hasattr(model.model, "load"):
    #             model.model.load(None)

    #     running_variables = {}
    #     for i, input in enumerate(inputs):
    #         if not isinstance(input, input_variables[i].type_class):
    #             raise Exception(
    #                 "Input type mismatch, expceted %s got %s"
    #                 % (
    #                     input_variables[i].type_class,
    #                     input.__class__,
    #                 )
    #             )
    #         running_variables[input_variables[i].local_id] = input

    #     for node in self.nodes:
    #         node_inputs: List[Variable] = []
    #         node_outputs: List[Variable] = []
    #         node_function: Function = None

    #         for _node_input in node.inputs:
    #             print("node inputs local id",_node_input.local_id)
    #             for variable in self.variables:
    #                 print("var local id",variable.local_id)
    #                 if (
    #                     variable.local_id == _node_input.local_id
    #                     or variable.belongs_to == self.name
    #                 ):
    #                     node_inputs.append(variable)
    #                     break
    #         for _node_output in node.outputs:
    #             if _node_output.belongs_to == self.name:
    #                 node_outputs.append(variable)
    #             for variable in self.variables:
    #                 # print(
    #                 #     "var id",
    #                 #     variable.local_id,
    #                 #     variable.is_output,
    #                 #     variable.belongs_to,
    #                 # )
    #                 if (
    #                     variable.local_id == _node_output.local_id
    #                     or variable.is_output
    #                 ):
    #                     node_outputs.append(variable)
    #                     break
    #         for function in self.functions:
    #             if function.local_id == node.function.local_id:
    #                 node_function = function
    #                 break

    #         function_inputs = []
    #         for _input in node_inputs:
    #             function_inputs.append(running_variables[_input.local_id])

    #         if (
    #             hasattr(node.function, "class_instance")
    #             and node_function.class_instance is not None
    #         ):
    #             output = node_function.function(
    #                 node_function.class_instance, *function_inputs
    #             )
    #         else:
    #             output = node_function.function(*function_inputs)
    #         running_variables[node_outputs[0].local_id] = output

    #     return_variables = []
    #     for output_variable in self.outputs:
    #         return_variables.append(
    #             running_variables[output_variable.local_id]
    #         )

    #     return return_variables

    def run_node(self, node: GraphNode, *inputs: Any) -> Any:
        input_variables = node.inputs
        # input_variables: List[Variable] = [
        #     var
        #     for var in self.variables
        #     if var.is_input and var.belongs_to == self.name
        # ]
        if len(inputs) != len(input_variables):
            raise Exception(
                "Mismatch of number of inputs, expecting %u got %s"
                % (len(input_variables), len(inputs))
            )
        # for v in input_variables:
        #     print(v.local_id)

        # check and store actual values to run with
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
        f_inputs: List[Variable] = []
        for input in node.inputs:
            f_inputs.append(running_variables[input.local_id])

        node_exec = node.function
        return node_exec.function(*f_inputs)

    def run(self, *inputs: Any) -> Tuple[Any, ...]:
        print(self.nodes)
        if self.nodes is None or len(self.nodes) == 0:
            return inputs
        node = self.nodes.pop()
        results = self.run_node(node, inputs)
        return self.run(*results)

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
    def from_schema(cls, schema: PipelineGet) -> Graph:
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
            # variables=variables,
            # functions=functions,
            # outputs=outputs,
            nodes=nodes,
        )

        return remade_graph
