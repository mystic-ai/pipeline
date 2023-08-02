import tempfile
from typing import Any, Iterable, List, Optional

from cloudpickle import dumps
from dill import loads
from pydantic import BaseModel

from pipeline.cloud.schemas.pipelines import IOVariable
from pipeline.cloud.schemas.runs import RunIOType
from pipeline.objects.function import Function
from pipeline.objects.model import Model
from pipeline.util import dump_object, generate_id


class VariableException(Exception):
    ...


class Variable:
    local_id: str
    remote_id: str

    name: str

    type_class: Any

    is_input: bool
    is_output: bool

    def __init__(
        self,
        type_class: Any,
        *,
        is_input: bool = True,
        is_output: bool = False,
        local_id: str = None,
        # default: Any | None = None, # TODO: Implement default values
        title: str | None = None,
        description: str | None = None,
        examples: list[Any] | None = None,
        gt: int | None = None,
        ge: int | None = None,
        lt: int | None = None,
        le: int | None = None,
        multiple_of: int | None = None,
        allow_inf_nan: bool | None = None,
        max_digits: int | None = None,
        decimal_places: int | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        choices: list[Any] | None = None,
        dict_schema: BaseModel | None = None,
    ):
        from pipeline.objects.pipeline import Pipeline

        self.type_class = type_class
        self.is_input = is_input
        self.is_output = is_output

        self.title = title
        self.description = description
        self.examples = examples
        self.gt = gt
        self.ge = ge
        self.lt = lt
        self.le = le
        self.multiple_of = multiple_of
        self.allow_inf_nan = allow_inf_nan
        self.max_digits = max_digits
        self.decimal_places = decimal_places
        self.min_length = min_length
        self.max_length = max_length
        self.choices = choices
        self.dict_schema = dict_schema.schema() if dict_schema is not None else None

        if not Pipeline._pipeline_context_active:
            raise Exception("Cant add a variable when not defining a pipeline")

        Pipeline._current_pipeline.variables.append(self)

        self.local_id = generate_id(10) if not local_id else local_id

    def validate_variable(self, value: Any) -> None:
        # if not isinstance(value, self.type_class):
        #     return

        if self.choices is not None:
            if value not in self.choices:
                raise VariableException(
                    f"Value is not in the choices of {self.choices}"
                )

        if self.gt is not None:
            if value <= self.gt:
                raise VariableException(f"Value is not greater than {self.gt}")

        if self.ge is not None:
            if value < self.ge:
                raise VariableException(
                    f"Value is not greater than or equal to {self.ge}"
                )

        if self.lt is not None:
            if value >= self.lt:
                raise VariableException(f"Value is not less than {self.lt}")

        if self.le is not None:
            if value > self.le:
                raise VariableException(f"Value is not less than or equal to {self.le}")

        if self.multiple_of is not None:
            if value % self.multiple_of != 0:
                raise VariableException(
                    f"Value is not a multiple of {self.multiple_of}"
                )

        if self.allow_inf_nan is not None:
            if not self.allow_inf_nan and (
                value == float("inf") or value == float("-inf") or value == float("nan")
            ):
                raise VariableException("Value is not allowed to be infinity or nan")

        if self.max_digits is not None:
            if len(str(value)) > self.max_digits:
                raise VariableException(f"Value has more than {self.max_digits} digits")

        if self.decimal_places is not None:
            if len(str(value).split(".")[1]) > self.decimal_places:
                raise VariableException(
                    f"Value has more than {self.decimal_places} decimal places"
                )

        if self.min_length is not None:
            if len(str(value)) < self.min_length:
                raise VariableException(
                    f"Value has less than {self.min_length} characters"
                )

        if self.max_length is not None:
            if len(str(value)) > self.max_length:
                raise VariableException(
                    f"Value has more than {self.max_length} characters"
                )

    def to_io_schema(self) -> IOVariable:
        return IOVariable(
            run_io_type=RunIOType.from_object(self.type_class),
            title=self.title,
            description=self.description,
            examples=self.examples,
            gt=self.gt,
            ge=self.ge,
            lt=self.lt,
            le=self.le,
            multiple_of=self.multiple_of,
            allow_inf_nan=self.allow_inf_nan,
            max_digits=self.max_digits,
            decimal_places=self.decimal_places,
            min_length=self.min_length,
            max_length=self.max_length,
            choices=self.choices,
            dict_schema=self.dict_schema,
        )


class PipelineFile(Variable):
    path: str

    def __init__(
        self,
        *,
        path: str = None,
        title: str = None,
        local_id: str = None,
    ) -> None:
        super().__init__(
            type_class=self.__class__,
            is_input=False,
            is_output=False,
            title=title,
            local_id=local_id,
        )
        self.path = path

    @classmethod
    def from_object(
        cls,
        obj: Any,
        modules: Optional[List[str]] = None,
    ):
        temp_file = tempfile.NamedTemporaryFile(delete=False)

        bytes = dump_object(obj, modules=modules)
        temp_file.write(bytes)
        temp_file.seek(0)

        return cls(
            path=temp_file.name,
            name=temp_file.name,
        )


class Stream(Variable, Iterable):
    ...


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


class Graph:
    local_id: str
    remote_id: str

    name: str

    functions: List[Function]
    variables: List[Variable]

    outputs: List[Variable]

    nodes: List[GraphNode]

    models: List[Model]

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
        # Flag set when all functions with the on_startup field have run
        self._has_run_startup = False

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
                and getattr(node_function.function, "__has_run__", False)
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

            if getattr(node_function.function, "__run_once__", False):
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
                if var.remote_id is not None and not var.path:
                    raise Exception(
                        "Must call PipelineCloud().download_remotes(...) on "
                        "remote PipelineFiles"
                    )

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

            if len(node.outputs) > 1:
                if len(node.outputs) == len(output) == len(node_outputs):
                    number_of_outputs = len(output)
                    for i in range(number_of_outputs):
                        running_variables[node_outputs[i].local_id] = output[i]
                else:
                    raise Exception(
                        "Mismatch in number of outputs:"
                        f"{len(node.outputs)}/{len(output)}/{len(node_outputs)}"
                    )
            else:
                running_variables[node_outputs[0].local_id] = output

            if not getattr(node_function.function, "__has_run__", False):
                node_function.function.__has_run__ = True

        return_variables = []

        for output_variable in self.outputs:
            return_variables.append(running_variables[output_variable.local_id])

        return return_variables

    def save(self, save_path):
        with open(save_path, "wb") as save_file:
            save_file.write(dumps(self))

    @classmethod
    def load(cls, load_path):
        with open(load_path, "rb") as load_file:
            return loads(load_file.read())
