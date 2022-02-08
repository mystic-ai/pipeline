from typing import List

from dill import dumps, loads

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

    functions: List[Function]
    variables: List[Variable]

    outputs: List[Variable]

    nodes: List[GraphNode]

    models: List[Model]
    # TODO: Add generic objects (e.g. Model) to be included in the graph

    def __init__(
        self,
        *,
        name: str = "",
        variables: List[Variable] = None,
        functions: List[Function] = None,
        outputs: List[Variable] = None,
        nodes: List[GraphNode] = None,
        models: List[Model] = None,
    ):
        self.name = name
        self.local_id = generate_id(10)

        self.variables = variables if variables is not None else []
        self.functions = functions if functions is not None else []
        self.outputs = outputs if outputs is not None else []
        self.nodes = nodes if nodes is not None else []
        self.models = models if models is not None else []

    def run(self, *inputs):
        input_variables: List[Variable] = [
            var for var in self.variables if var.is_input
        ]

        # TODO: Add generic object loading

        if len(inputs) != len(input_variables):
            raise Exception(
                "Mismatch of number of inputs, expecting %u got %s"
                % (len(input_variables), len(inputs))
            )

        for model in self.models:
            # TODO check dir of model.model
            print(
                "Attempt @ load:%s,%s,%s\n%s"
                % (model, model.model, model.model.model, dir(model.model))
            )
            if hasattr(model.model, "load"):
                print("Loading model (%s)" % model.local_id)
                model.model.load()
            else:
                raise Exception("Model load not found")

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

        for node in self.nodes:

            node_inputs: List[Variable] = []
            node_outputs: List[Variable] = []
            node_function: Function = None

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

            for function in self.functions:
                if function.local_id == node.function.local_id:
                    node_function = function
                    break

            function_inputs = []
            for _input in node_inputs:
                function_inputs.append(running_variables[_input.local_id])
            print("Running#2")

            print("Dir of node_function:\n%s\n" % dir(node_function))
            print("node_function.function:\n%s\n" % (node_function.function))
            print("Dir of node_function.function:\n%s\n" % dir(node_function.function))
            print(
                "Dir of node_function.class_instance:\n%s\n"
                % dir(node_function.class_instance)
            )
            if node_function.function == None:
                raise Exception(
                    "Node function is none (id:%s)" % node.function.local_id
                )

            if (
                hasattr(node_function, "class_instance")
                and node_function.class_instance is not None
            ):
                output = node_function.function(
                    node_function.class_instance, *function_inputs
                )
            else:
                output = node_function.function(*function_inputs)

            running_variables[node_outputs[0].local_id] = output

        return_variables = []

        for output_variable in self.outputs:
            return_variables.append(running_variables[output_variable.local_id])

        return return_variables

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
        print(models[0].model.local_id)
        # Rebind functions -> models
        update_functions = []
        for _func in functions:
            print(_func)
            print(_func.class_instance)
            print(_func.class_instance.__pipeline_model__)
            if hasattr(_func.class_instance, "__pipeline_model__"):
                model = _func.class_instance
                is_bound = False
                for _model in models:
                    if _model.model.local_id == model.local_id:
                        """
                        if as_name is None:
                            as_name = func.__name__
                        bound_method = func.__get__(instance, instance.__class__)
                        setattr(instance, as_name, bound_method)
                        return bound_method
                        """
                        bound_method = _func.function.__get__(
                            _model.model, _model.model.__class__
                        )
                        setattr(_model.model, _func.function.__name__, bound_method)
                        is_bound = True
                        _func.class_instance = model
                if not is_bound:
                    raise Exception(
                        "Did not find a class to bind for model (local_id:%s)"
                        % model.local_id
                    )
            else:
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
