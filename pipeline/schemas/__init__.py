import string
import random
from pydantic import BaseModel
from typing import List, Optional, Callable, Any


class PipelineVariableSchema(BaseModel):
    variable_type: Any
    variable_name: str
    is_input: bool = False
    is_output: bool = False

    def __init__(self, **kwargs):
        if not "variable_name" in kwargs:
            kwargs["variable_name"] = "".join(
                random.choice(string.ascii_lowercase) for i in range(10)
            )
        super().__init__(**kwargs)

    def json(self):
        # print("coming in")
        return self.dict()

    def dict(self, *args, **kwargs):
        # print("Things")
        return {
            "variable_type": str(self.variable_type.__name__),
            "variable_name": self.variable_name,
            "is_input": self.is_input,
            "is_output": self.is_output,
        }


class PipelineInputVariableSchema(BaseModel):
    variable: PipelineVariableSchema


class PipelineOutputVariableSchema(BaseModel):
    variable: PipelineVariableSchema


class PipelineModel(BaseModel):
    model: Any
    name: str

    def __init__(self,*args, **kwargs):
        if not "name" in kwargs:
            kwargs["name"] = "".join(
                random.choice(string.ascii_lowercase) for i in range(10)
            )
        super().__init__(**kwargs)


class PipelineFunctionSchema(BaseModel):
    inputs: dict
    name: str
    hash: Optional[str]
    function: Callable
    bound_class: Optional[Any]

    def dict(self, *args, **kwargs):
        return {
            "inputs": {
                input_name: self.inputs[input_name].__class__.__name__
                for input_name in self.inputs
            },
            "name": self.name,
            "hash": self.hash,
            # "function": self.function,
        }


class PipelineGraphNodeSchema(BaseModel):
    pipeline_function: PipelineFunctionSchema
    inputs: List[PipelineVariableSchema]
    output: PipelineVariableSchema


class PipelineGraph(BaseModel):
    inputs: List[PipelineInputVariableSchema]
    variables: List[PipelineVariableSchema]
    outputs: List[PipelineOutputVariableSchema]
    graph_nodes: List[PipelineGraphNodeSchema]
    models: List[PipelineModel]
