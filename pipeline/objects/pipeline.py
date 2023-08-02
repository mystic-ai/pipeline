from pipeline.objects.function import Function
from pipeline.objects.graph import Graph, GraphNode, Variable


class Pipeline:
    _current_pipeline: Graph
    _pipeline_context_active: bool = False

    def __init__(
        self,
    ):
        ...

    def __enter__(self):
        Pipeline._pipeline_context_active = True

        Pipeline._current_pipeline = Graph()

        return self

    def __exit__(self, type, value, traceback):
        Pipeline._pipeline_context_active = False

    def output(self, *outputs: Variable) -> None:
        for _output in outputs:
            variable_index = Pipeline._current_pipeline.variables.index(_output)
            if variable_index != -1:
                Pipeline._current_pipeline.variables[variable_index].is_output = True

                Pipeline._current_pipeline.outputs.append(
                    Pipeline._current_pipeline.variables[variable_index]
                )
                for variable in Pipeline._current_pipeline.variables:
                    if variable.local_id == _output.local_id:
                        variable.is_output = True
                        break

    def get_pipeline(self) -> Graph:
        return Pipeline._current_pipeline

    @staticmethod
    def add_function(function: Function) -> None:
        if Pipeline._pipeline_context_active:
            Pipeline._current_pipeline.functions.append(function)
        else:
            raise Exception("Cant add a function when not defining a pipeline!")

    @staticmethod
    def add_graph_node(graph_node: GraphNode) -> None:
        if Pipeline._pipeline_context_active:
            Pipeline._current_pipeline.nodes.append(graph_node)
        else:
            raise Exception("Cant add a node when not defining a pipeline!")
