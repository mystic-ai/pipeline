from typing import List

from pipeline.objects.function import Function
from pipeline.objects.variable import Variable
from pipeline.schemas.pipeline import PipelineGraphNode
from pipeline.util import generate_id


class GraphNode:
    local_id: str
    function: Function
    inputs: List[Variable] = []
    outputs: List[Variable] = []

    def __init__(self, function, inputs, outputs, *, local_id=None):
        self.function = function
        self.inputs = inputs
        self.outputs = outputs

        self.local_id = generate_id(10) if local_id is None else local_id

    def to_create_schema(self) -> PipelineGraphNode:
        return PipelineGraphNode(
            local_id=self.local_id,
            function=self.function.local_id,
            inputs=[_var.local_id for _var in self.inputs],
            outputs=[_var.local_id for _var in self.outputs],
        )
