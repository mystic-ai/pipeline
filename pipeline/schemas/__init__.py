import os
import string
import random
import inspect

from dill import dumps

from hashlib import sha256

from pydantic import BaseModel

from typing import List, Optional, Callable, Any


class PipelineVariableSchema(BaseModel):
    variable_type: Any
    variable_type_file_path: Optional[str]
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
            "variable_type_file_path": self.variable_type_file_path,
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
    hash: Optional[str]
    model_file_name: Optional[str]

    def __init__(self, *args, **kwargs):
        if not "name" in kwargs:
            kwargs["name"] = "".join(
                random.choice(string.ascii_lowercase) for i in range(10)
            )
        super().__init__(**kwargs)

    def dict(self, *args, **kwargs):
        return {
            "name": self.name,
            "hash": self.hash,
            "model_file_name": self.model_file_name,
        }


class PipelineFunctionSchema(BaseModel):
    inputs: dict
    name: str
    hash: Optional[str]
    function: Optional[Callable]
    function_file_name: Optional[str]
    bound_class: Optional[Any]
    bound_class_file_name: Optional[str]

    def dict(self, *args, **kwargs):
        return {
            "inputs": {
                input_name: self.inputs[input_name].__class__.__name__
                for input_name in self.inputs
            },
            "name": self.name,
            "hash": self.hash,
            "function_file_name": self.function_file_name,
            "bound_class_file_name": self.bound_class_file_name,
        }


class PipelineGraphNodeSchema(BaseModel):
    pipeline_function: PipelineFunctionSchema
    inputs: List[PipelineVariableSchema]
    output: PipelineVariableSchema


class PipelineGraph(BaseModel):
    name: str
    inputs: List[PipelineInputVariableSchema]
    variables: List[PipelineVariableSchema]
    outputs: List[PipelineOutputVariableSchema]
    graph_nodes: List[PipelineGraphNodeSchema]
    models: List[PipelineModel]

    def save(self, save_dir, tar=False, overwrite=True):

        base_save_path = os.path.join(save_dir, "%s" % self.name)

        if not overwrite and len(os.listdir(base_save_path)) > 0:
            raise Exception("Save directory not empty.")
        elif overwrite and len(os.listdir(base_save_path)) > 0:
            for file in os.listdir(base_save_path):
                os.remove(os.path.join(base_save_path, file))

        print("Saving pipeline in: '%s'" % base_save_path)
        os.makedirs(base_save_path, exist_ok=True)

        # Save functions
        for node in self.graph_nodes:
            pipeline_function = node.pipeline_function
            function_data = dumps(pipeline_function.function)
            function_source = inspect.getsource(pipeline_function.function)
            function_hash = sha256(function_source.encode()).hexdigest()
            save_file_name = function_hash + ".fn"

            # Update node info
            node.pipeline_function.hash = function_hash
            node.pipeline_function.function_file_name = save_file_name

            with open(
                os.path.join(base_save_path, save_file_name), "wb"
            ) as node_function_file:
                node_function_file.write(function_data)
            if node.pipeline_function.bound_class != None:
                bound_class_data = dumps(node.pipeline_function.bound_class)
                # bound_class_source = inspect.getsource(
                #    node.pipeline_function.bound_class
                # )
                bound_class_hash = sha256(bound_class_data).hexdigest()
                bound_class_file_name = bound_class_hash + ".cls"
                node.pipeline_function.bound_class_file_name = bound_class_file_name
                with open(
                    os.path.join(base_save_path, bound_class_file_name), "wb"
                ) as bound_class_file:
                    bound_class_file.write(bound_class_data)
        # Save models
        for model in self.models:
            model_data = dumps(model.model)
            model_hash = sha256(model_data).hexdigest()
            model.hash = model_hash
            model_file_name = model_hash + ".mdl"
            model.model_file_name = model_file_name
            with open(
                os.path.join(base_save_path, model_file_name), "wb"
            ) as model_file:
                model_file.write(model_data)

        # Save variables
        for variable in self.variables:
            type_data = dumps(variable.variable_type)
            type_hash = sha256(type_data).hexdigest()
            type_file_name = type_hash + ".typ"
            variable.variable_type_file_path = type_file_name
            with open(os.path.join(base_save_path, type_file_name), "wb") as type_file:
                type_file.write(type_data)
        for variable in self.inputs:
            type_data = dumps(variable.variable.variable_type)
            type_hash = sha256(type_data).hexdigest()
            type_file_name = type_hash + ".typ"
            variable.variable.variable_type_file_path = type_file_name
            with open(os.path.join(base_save_path, type_file_name), "wb") as type_file:
                type_file.write(type_data)
        for variable in self.outputs:
            type_data = dumps(variable.variable.variable_type)
            type_hash = sha256(type_data).hexdigest()
            type_file_name = type_hash + ".typ"
            variable.variable.variable_type_file_path = type_file_name
            with open(os.path.join(base_save_path, type_file_name), "wb") as type_file:
                type_file.write(type_data)

        with open(os.path.join(base_save_path, "config.json"), "w") as config_file:
            config_file.write(self.json())
