import io

from typing import List

from pipeline.api.call import post

from pipeline.schemas.pipeline import (
    PipelineCreate,
    PipelineGet,
    PipelineGraphNode,
    PipelineVariableGet,
)

from pipeline.objects.graph import Graph

from pipeline.util import python_object_to_hex

from pipeline.api.file import upload_file
from pipeline.api.function import upload_function

from pipeline.util.logging import _print


def upload_pipeline(new_pipeline_graph: Graph) -> PipelineGet:

    new_name = new_pipeline_graph.name
    _print("Uploading functions")
    new_functions = [
        upload_function(_function) for _function in new_pipeline_graph.functions
    ]

    new_variables: List[PipelineVariableGet] = []
    _print("Uploading variables")
    for _var in new_pipeline_graph.variables:
        _var_type_file = upload_file(
            io.BytesIO(python_object_to_hex(_var.type_class).encode()), "/"
        )
        _var_schema = PipelineVariableGet(
            local_id=_var.local_id,
            name=_var.name,
            type_file=_var_type_file,
            is_input=_var.is_input,
            is_output=_var.is_output,
        )
        new_variables.append(_var_schema)

    new_graph_nodes = [_node.to_create_schema() for _node in new_pipeline_graph.nodes]
    new_outputs = [_output.local_id for _output in new_pipeline_graph.outputs]

    pipeline_create_schema = PipelineCreate(
        name=new_name,
        variables=new_variables,
        functions=new_functions,
        graph_nodes=new_graph_nodes,
        outputs=new_outputs,
    )
    _print("Uploading pipeline graph")
    request_result = post("/v2/pipelines", pipeline_create_schema.dict())

    return PipelineGet.parse_obj(request_result)
