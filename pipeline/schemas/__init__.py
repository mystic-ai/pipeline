import os
import string
import random
import inspect

from dill import dumps

from hashlib import sha256

from pydantic import BaseModel

from typing import List, Optional, Callable, Any

from pipeline.objects import PipelineFunction


class PipelineVariableSchema(BaseModel):
    id: str

    variable_type: Any
    variable_type_file_path: Optional[str]
    variable_name: str
    is_input: bool = False
    is_output: bool = False
    variable_name: str

    def __init__(self, **kwargs):
        if not "variable_name" in kwargs:
            kwargs["variable_name"] = "".join(
                random.choice(string.ascii_lowercase) for i in range(10)
            )
        kwargs["id"] = "".join(random.choice(string.ascii_lowercase) for i in range(20))
        super().__init__(**kwargs)

    def json(self):
        # print("coming in")
        return self.dict()

    def dict(self, *args, **kwargs):
        # print("Things")
        return {
            "id": self.id,
            "variable_type": str(self.variable_type.__name__),
            "variable_type_file_path": self.variable_type_file_path,
            "variable_name": self.variable_name,
            "is_input": self.is_input,
            "is_output": self.is_output,
        }


class PipelineModel(BaseModel):
    id: str
    remote_id: Optional[str]

    model: Any
    name: str
    hash: Optional[str]
    model_file_name: Optional[str]

    def __init__(self, *args, **kwargs):
        kwargs["id"] = "".join(random.choice(string.ascii_lowercase) for i in range(20))
        if not "name" in kwargs:
            kwargs["name"] = "".join(
                random.choice(string.ascii_lowercase) for i in range(10)
            )
        super().__init__(**kwargs)

    def dict(self, *args, **kwargs):
        return {
            "id": self.id,
            "remote_id": self.remote_id,
            "name": self.name,
            "hash": self.hash,
            "model_file_name": self.model_file_name,
        }


"""
class PipelineFunctionSchema(BaseModel):
    id: str
    remote_id: Optional[str]

    name: str
    hash: Optional[str]
    inputs: dict

    function_hex: str
    function_source: str

    function: Optional[Callable]
    function_file_name: Optional[str]

    bound_class: Optional[Any]
    bound_class_file_name: Optional[str]

    def __init__(self, **kwargs):
        kwargs["id"] = "".join(random.choice(string.ascii_lowercase) for i in range(20))

        function_source = inspect.getsource(kwargs["function"])
        kwargs["hash"] = sha256(function_source.encode()).hexdigest()
        super().__init__(**kwargs)

    def dict(self, *args, **kwargs):
        return {
            "id": self.id,
            "remote_id": self.remote_id,
            "inputs": {
                input_name: self.inputs[input_name].__name__
                for input_name in self.inputs
            },
            "name": self.name,
            "hash": self.hash,
            "function_file_name": self.function_file_name,
            "bound_class_file_name": self.bound_class_file_name,
        }
"""


class PipelineGraphNodeSchema(BaseModel):
    id: str
    pipeline_function: str
    inputs: List[str]
    outputs: List[str]

    def __init__(self, *args, **kwargs):
        kwargs["id"] = "".join(random.choice(string.ascii_lowercase) for i in range(20))
        super().__init__(*args, **kwargs)


class PipelineGraph(BaseModel):
    id: str
    remote_id: Optional[str]

    name: str

    functions: List[PipelineFunction]
    variables: List[PipelineVariableSchema]

    graph_nodes: List[PipelineGraphNodeSchema]

    models: List[PipelineModel]

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, *args, **kwargs):
        kwargs["id"] = "".join(random.choice(string.ascii_lowercase) for i in range(20))
        super().__init__(*args, **kwargs)

    def run(self, *inputs):

        input_variables: List[PipelineVariableSchema] = [
            var for var in self.variables if var.is_input
        ]
        output_variables: List[PipelineVariableSchema] = [
            var for var in self.variables if var.is_output
        ]

        for model in self.models:
            if hasattr(model.model, "load"):
                model.model.load(None)

        if len(inputs) != len(input_variables):
            raise Exception(
                "Mismatch of number of inputs, expecting %u got %s"
                % (len(input_variables), len(inputs))
            )
        running_variables = {}
        for i, input in enumerate(inputs):
            if not isinstance(input, input_variables[i].variable_type):
                raise Exception(
                    "Input type mismatch, expceted %s got %s"
                    % (
                        input_variables[i].variable_type,
                        input.__class__,
                    )
                )
            running_variables[input_variables[i].variable_name] = input

        for node in self.graph_nodes:

            node_inputs: List[PipelineVariableSchema] = []
            node_outputs: List[PipelineVariableSchema] = []
            node_function: PipelineFunction = None

            for _node_input in node.inputs:
                for variable in self.variables:
                    if variable.variable_name == _node_input:
                        node_inputs.append(variable)
                        break
            for _node_output in node.outputs:
                for variable in self.variables:
                    if variable.variable_name == _node_output:
                        node_outputs.append(variable)
                        break

            for function in self.functions:
                if function.name == node.pipeline_function:
                    node_function = function
                    break

            function_inputs = []
            for _input in node_inputs:
                function_inputs.append(running_variables[_input.variable_name])

            if (
                hasattr(node.pipeline_function, "bound_class")
                and node_function.bound_class != None
            ):
                output = node_function.function(
                    node_function.bound_class, *function_inputs
                )
            else:
                output = node_function.function(*function_inputs)

            running_variables[node_outputs[0].variable_name] = output

        return_variables = []
        for output_variable in output_variables:
            return_variables.append(running_variables[output_variable.variable_name])

        return return_variables

    def save(self, save_dir, tar=False, overwrite=True):

        base_save_path = os.path.join(save_dir, "%s" % self.name)
        if os.path.exists(base_save_path):
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
            if (
                hasattr(node.pipeline_function, "bound_class")
                and node.pipeline_function.bound_class != None
            ):
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

        with open(os.path.join(base_save_path, "config.json"), "w") as config_file:

            config_file.write(self.json())
