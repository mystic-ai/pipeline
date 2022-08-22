from typing import List

from cloudpickle import dumps
from dill import loads

from pipeline.objects.function import Function
from pipeline.objects.graph_node import GraphNode
from pipeline.objects.model import Model
from pipeline.objects.variable import PipelineFile, Variable
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

    compute_type: str
    min_gpu_vram_mb: int

    def __init__(
        self,
        *,
        name: str = "",
        variables: List[Variable] = None,
        functions: List[Function] = None,
        outputs: List[Variable] = None,
        nodes: List[GraphNode] = None,
        models: List[Model] = None,
        compute_type: str = "gpu",
        min_gpu_vram_mb: int = None,
    ):
        self.name = name
        self.local_id = generate_id(10)

        self.variables = variables if variables is not None else []
        self.functions = functions if functions is not None else []
        self.outputs = outputs if outputs is not None else []
        self.nodes = nodes if nodes is not None else []
        self.models = models if models is not None else []
        # Flag set when all functions with the on_startup field have run
        self._has_run_startup = False
        self.compute_type = compute_type
        self.min_gpu_vram_mb = min_gpu_vram_mb

    def _startup(self):
        if self._has_run_startup:
            return

        startup_variables = {}

        for var in self.variables:
            # At the moment only the PipelineFile variable can be used on startup
            if isinstance(var, PipelineFile):
                startup_variables[var.local_id] = var

        for node in self.nodes:

            node_inputs: List[Variable] = []
            node_function: Function = None

            for function in self.functions:
                if function.local_id == node.function.local_id:
                    node_function = function
                    break
            if (
                getattr(node_function.function, "__run_once__", False)
                and not getattr(node_function.function, "__has_run__", False)
            ) or not getattr(node_function.function, "__on_startup__", False):
                continue

            for _node_input in node.inputs:
                for variable in self.variables:
                    if variable.local_id == _node_input.local_id:
                        node_inputs.append(variable)
                        break

            function_inputs = []
            for _input in node_inputs:
                function_inputs.append(startup_variables[_input.local_id])

            if node_function.function is None:
                raise Exception(
                    "Node function is None (id:%s)" % node.function.local_id
                )

            if getattr(node_function, "class_instance", None) is not None:
                node_function.function(node_function.class_instance, *function_inputs)
            else:
                node_function.function(*function_inputs)

            if getattr(node_function.function, "__has_run__", False):
                node_function.function.__has_run__ = True

        self._has_run_startup = True

    def run(self, *inputs):
        input_variables: List[Variable] = [
            var for var in self.variables if var.is_input
        ]

        if len(inputs) != len(input_variables):
            raise Exception(
                "Mismatch of number of inputs, expecting %u got %s"
                % (len(input_variables), len(inputs))
            )

        self._startup()

        running_variables = {}

        # Add all PipelineFile's to the running variables
        for var in self.variables:
            if isinstance(var, PipelineFile):
                running_variables[var.local_id] = var

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

            for function in self.functions:
                if function.local_id == node.function.local_id:
                    node_function = function
                    break

            if getattr(node_function.function, "__run_once__", False) and getattr(
                node_function.function, "__has_run__", False
            ):
                continue

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

            function_inputs = []
            for _input in node_inputs:
                function_inputs.append(running_variables[_input.local_id])

            if node_function.function is None:
                raise Exception(
                    "Node function is none (id:%s)" % node.function.local_id
                )

            if getattr(node_function, "class_instance", None) is not None:
                output = node_function.function(
                    node_function.class_instance, *function_inputs
                )
            else:
                output = node_function.function(*function_inputs)

            running_variables[node_outputs[0].local_id] = output

            if not getattr(node_function.function, "__has_run__", False):
                node_function.function.__has_run__ = True

        return_variables = []

        for output_variable in self.outputs:
            return_variables.append(running_variables[output_variable.local_id])

        return return_variables

    def _update_function_local_id(self, old_id: str, new_id: str) -> None:
        for func in self.functions:
            if func.local_id == old_id:
                func.local_id = new_id
                return
        raise Exception("Function with local_id:%s not found" % old_id)

    @classmethod
    def from_schema(cls, schema: PipelineGet):
        variables = [Variable.from_schema(_var) for _var in schema.variables]
        functions = [Function.from_schema(_func) for _func in schema.functions]
        models = [Model.from_schema(_model) for _model in schema.models]

        # Rebind functions -> models
        update_functions = []
        for _func in functions:
            if hasattr(_func.class_instance, "__pipeline_model__"):
                model = _func.class_instance
                is_bound = False
                for _model in models:
                    if _model.model.local_id == model.local_id:
                        bound_method = _func.function.__get__(
                            _model.model, _model.model.__class__
                        )
                        setattr(_model.model, _func.function.__name__, bound_method)
                        is_bound = True
                        _func.class_instance = _model.model
                if not is_bound:
                    raise Exception(
                        "Did not find a class to bind for model (local_id:%s)"
                        % model.local_id
                    )
            elif _func.class_instance is not None:
                raise Exception(
                    "Incorrect bound class:%s\ndir:%s"
                    % (_func.class_instance, dir(_func.class_instance))
                )
            update_functions.append(_func)
        functions = update_functions

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
            models=models,
        )

        return remade_graph

    def save(self, save_path):
        with open(save_path, "wb") as save_file:
            save_file.write(dumps(self))

    @classmethod
    def load(cls, load_path):
        with open(load_path, "rb") as load_file:
            return loads(load_file.read())
