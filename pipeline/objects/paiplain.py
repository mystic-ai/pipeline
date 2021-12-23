from typing import Any

from pipeline.objects.function import Function
from pipeline.objects.graph import Graph, GraphNode
from pipeline.objects.variable import Variable


class Paiplain:
    graph: Graph
    pipeline_context_name: str = None

    def __init__(self, new_pipeline_name):
        self.pipeline_context_name = new_pipeline_name

    def output(self, *outputs: Variable) -> None:
        for _output in outputs:
            if _output in self.graph.variables:
                variable_index = self.graph.variables.index(_output)
                self.graph.variables[variable_index].is_output = True

                self.graph.outputs.append(self.graph.variables[variable_index])
                for variable in self.graph.variables:
                    if variable.local_id == _output.local_id:
                        variable.is_output = True
                        break

    def get_graph(self) -> Graph:
        return self.graph

    def add_variable(
        self,
        type_class: Any,
        is_input: bool = False,
        is_output: bool = False,
        name: str = None,
        remote_id: str = None,
        local_id: str = None,
    ) -> Variable:
        variable = Variable(
            type_class,
            is_input=is_input,
            is_output=is_output,
            name=name,
            remote_id=remote_id,
            local_id=local_id,
            belongs_to=self.graph.name,
        )
        return self.graph.add_variable(variable)

    def add_function(self, function: Function) -> None:
        self.graph.add_function(function)

    def add_graph_node(self, graph_node: GraphNode) -> None:
        self.graph.add_node(graph_node)

    def run(self, *args) -> Any:
        return self.graph.run(*args)
