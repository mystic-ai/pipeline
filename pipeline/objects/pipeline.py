from typing import Any, Set, Union

from pipeline.api import PipelineCloud
from pipeline.objects.function import Function
from pipeline.objects.graph import Graph
from pipeline.objects.graph_node import GraphNode
from pipeline.objects.variable import Variable
from pipeline.schemas.pipeline import PipelineGet


class Pipeline:
    defined_pipelines = {}

    _current_pipeline: Graph
    _pipeline_context_active: bool = False
    _pipeline_context_name: str = None
    _api: PipelineCloud = None
    _compute_type: str = "gpu"
    _min_gpu_vram_mb: int = None

    def __init__(
        self,
        new_pipeline_name: str,
        api: PipelineCloud = None,
        compute_type: str = "gpu",
        min_gpu_vram_mb: int = None,
    ):
        self._pipeline_context_name = new_pipeline_name
        self._api = api or PipelineCloud()
        self._compute_type = compute_type
        self._min_gpu_vram_mb = min_gpu_vram_mb

    def __enter__(self):
        Pipeline._pipeline_context_active = True

        Pipeline._current_pipeline = Graph(
            name=self._pipeline_context_name,
            compute_type=self._compute_type,
            min_gpu_vram_mb=self._min_gpu_vram_mb,
        )

        return self

    def __exit__(self, type, value, traceback):
        Pipeline.defined_pipelines[
            Pipeline._current_pipeline.name
        ] = Pipeline._current_pipeline
        Pipeline._pipeline_context_active = False
        Pipeline._current_pipeline = None

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

    @staticmethod
    def get_pipeline(graph_name: str) -> Graph:
        """
        Get Graph representation of a pipeline.

            Parameters:
                    graph_name (str): name identifier of desired Pipeline

            Returns:
                    graph (Graph): Graph representation of Pipeline.
        """
        if graph_name in Pipeline.defined_pipelines:
            return Pipeline.defined_pipelines[graph_name]
        else:
            raise Exception("No Pipeline graph found with name '%s'" % graph_name)

    @staticmethod
    def add_variable(variable: Variable) -> None:
        if Pipeline._pipeline_context_active:

            if variable not in Pipeline._current_pipeline.variables:
                Pipeline._current_pipeline.variables.append(variable)
        else:
            raise Exception("Cant add a variable when not defining a pipeline!")

    @staticmethod
    def add_variables(*variables: Variable) -> None:
        for v in variables:
            Pipeline.add_variable(v)

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

    @staticmethod
    def run(graph_name: str, *inputs):
        graph = Pipeline.get_pipeline(graph_name)
        return graph.run(*inputs)

    @staticmethod
    def run_remote(id: Union[str, PipelineGet], data: Any):
        return Pipeline._api.run_pipeline(id, data)

    @staticmethod
    def upload(
        name: str, public: bool = False, description: str = "", tags: Set[str] = None
    ) -> PipelineGet:
        """
        Upload a Pipeline to the Cloud.

            Parameters:
                    name (str): name identifier of pipeline to be uploaded.
                    public (bool): If pipeline should be visible to public.
                        Defaults to False.
                    description (str): Description of the Pipeline.
                    tags (Set[str]): Set of tags for the pipeline. Eg: {"BERT", "NLP"}

            Returns:
                    pipeline (PipelineGet): Object representing uploaded pipeline.
        """
        graph = Pipeline.get_pipeline(name)
        return Pipeline._api.upload_pipeline(graph, public, description, tags)

    @staticmethod
    def api(api: PipelineCloud = None) -> PipelineCloud:
        Pipeline._api = api or PipelineCloud()
        return Pipeline._api
