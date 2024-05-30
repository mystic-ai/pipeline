import inspect
from pathlib import Path
from types import NoneType, UnionType
from typing import Any, Generic, Iterable, List, TypeVar, get_args
from urllib.parse import ParseResult, urlparse

import httpx
from cloudpickle import dumps, loads
from tqdm import tqdm

from pipeline.cloud.schemas.pipelines import IOVariable
from pipeline.cloud.schemas.runs import RunIOType
from pipeline.exceptions import RunInputException
from pipeline.objects.function import Function
from pipeline.objects.model import Model
from pipeline.util import generate_id


class InputSchema:
    def __init__(self, **kwargs):
        for key, value in self.__annotations__.items():
            validation_field = getattr(self, key, None)
            if not isinstance(validation_field, InputField):
                raise Exception("Must be InputField")

            if key not in kwargs and (
                "typing.Optional" in str(value) or isinstance(value, UnionType)
            ):
                setattr(self, key, validation_field.default)
                continue

            if key not in kwargs and not (
                "typing.Optional" in str(value) or isinstance(value, UnionType)
            ):
                raise Exception(
                    f"Missing value for '{key}', if you want to make it optional, use 'typing.Optional' or the pipe operator for example: 'int | None'"  # noqa
                )

            validation_field.validate(kwargs[key])
            setattr(self, key, kwargs[key])

    def __repr__(self) -> str:
        vars = ", ".join(
            [f"{key}={getattr(self, key)}" for key in self.__annotations__.keys()]
        )
        return f"InputSchema({vars})"

    @classmethod
    def to_schema(cls) -> List[dict]:
        output_list: List[IOVariable] = []

        items = dir(cls)

        for item, title in [
            (_field, item)
            for item in items
            if (_field := getattr(cls, item, None)) and isinstance(_field, InputField)
        ]:
            var_type = cls.__annotations__[title]

            if isinstance(var_type, UnionType) or "typing.Optional" in str(var_type):
                var_union_types = list(get_args(var_type))
                if len(var_union_types) > 2:
                    raise Exception("Only support Union of 2 types")
                if NoneType in var_union_types:
                    var_union_types.remove(NoneType)
                else:
                    raise Exception("Only support Union with None")
                var_type = var_union_types[0]

            output_list.append(
                item._to_io_schema(
                    _type=var_type,
                    _title=title,
                ).dict()
            )
        return output_list

    @classmethod
    def from_schema(cls, schemas: List[dict]) -> "InputSchema":
        new_input_schema = cls()
        new_input_schema.__annotations__ = {}
        for schema in schemas:
            input_field = InputField._from_io_schema(IOVariable(**schema))

            setattr(
                new_input_schema,
                schema.get("title"),
                input_field,
            )

            new_input_schema.__annotations__[schema.get("title")] = RunIOType.to_object(
                schema.get("run_io_type")
            )

        return new_input_schema

    def parse_dict(self, input_dict: dict) -> None:
        for item, title in [
            (_field, item)
            for item in dir(self)
            if (_field := getattr(self, item, None)) and isinstance(_field, InputField)
        ]:
            print(f"Assesing {title}")
            entered_value = input_dict.get(title, None)
            if item.default is None and entered_value is None:
                raise ValueError(f"Field {title} is not optional, no value entered")

            item.validate(entered_value)

            print(f"Now {title}={entered_value}")
            setattr(self, title, entered_value)

    def to_dict(self) -> dict:
        return {key: getattr(self, key) for key in self.__annotations__.keys()}


class A:
    pass


T: Any = A


class InputField(T):
    def __init__(
        self,
        default: Any | None = None,
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
        optional: bool | None = False,
    ):
        self.default = default
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
        self.optional = optional

        if default is not None:
            try:
                self.validate(self.default)
            except TypeError as e:
                raise TypeError(
                    f"Default value {default} is invalid for field {self.title}"
                ) from e

    def _to_io_schema(self, _type: Any, _title: str) -> IOVariable:
        return IOVariable(
            run_io_type=RunIOType.from_object(_type),
            title=_title,
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
            default=self.default,
            optional=self.optional,
        )

    @classmethod
    def _from_io_schema(cls, schema: IOVariable):
        return cls(
            default=schema.default,
            description=schema.description,
            examples=schema.examples,
            gt=schema.gt,
            ge=schema.ge,
            lt=schema.lt,
            le=schema.le,
            multiple_of=schema.multiple_of,
            allow_inf_nan=schema.allow_inf_nan,
            max_digits=schema.max_digits,
            decimal_places=schema.decimal_places,
            min_length=schema.min_length,
            max_length=schema.max_length,
            choices=schema.choices,
            title=schema.title,
            optional=schema.optional,
        )

    def validate(self, value: Any):
        if self.choices is not None:
            if value not in self.choices:
                raise TypeError(f"Value is not in the choices of {self.choices}")

        if self.gt is not None:
            if value <= self.gt:
                raise TypeError(f"Value is not greater than {self.gt}")

        if self.ge is not None:
            if value < self.ge:
                raise TypeError(f"Value is not greater than or equal to {self.ge}")

        if self.lt is not None:
            if value >= self.lt:
                raise TypeError(f"Value is not less than {self.lt}")

        if self.le is not None:
            if value > self.le:
                raise TypeError(f"Value is not less than or equal to {self.le}")

        if self.multiple_of is not None:
            if value % self.multiple_of != 0:
                raise TypeError(f"Value is not a multiple of {self.multiple_of}")

        if self.allow_inf_nan is not None:
            if not self.allow_inf_nan and (
                value == float("inf") or value == float("-inf") or value == float("nan")
            ):
                raise TypeError("Value is not allowed to be infinity or nan")

        if self.max_digits is not None:
            if len(str(value)) > self.max_digits:
                raise TypeError(f"Value has more than {self.max_digits} digits")

        if self.decimal_places is not None:
            if len(str(value).split(".")[1]) > self.decimal_places:
                raise TypeError(
                    f"Value has more than {self.decimal_places} decimal places"
                )

        if self.min_length is not None:
            if len(str(value)) < self.min_length:
                raise TypeError(f"Value has less than {self.min_length} characters")

        if self.max_length is not None:
            if len(str(value)) > self.max_length:
                raise TypeError(f"Value has more than {self.max_length} characters")


class Variable:
    local_id: str | None
    remote_id: str | None

    name: str | None

    type_class: Any

    is_input: bool
    is_output: bool

    def __init__(
        self,
        type_class: Any,
        *,
        default: Any | None = None,
        is_input: bool = True,
        is_output: bool = False,
        local_id: str | None = None,
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
        allow_out_of_context_creation: bool = False,
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
        self.default = default
        self.dict_schema = (
            type_class.to_schema()
            if inspect.isclass(type_class)
            and (
                issubclass(type_class, InputSchema)
                or isinstance(type_class, InputSchema)
            )
            else None
        )
        if not Pipeline._pipeline_context_active and not allow_out_of_context_creation:
            raise Exception("Cant add a variable when not defining a pipeline")

        if Pipeline._pipeline_context_active:
            Pipeline._current_pipeline.variables.append(self)

        self.local_id = generate_id(10) if not local_id else local_id

    def validate_variable(self, value: Any) -> None:
        # if not isinstance(value, self.type_class):
        #     return

        if self.choices is not None:
            if value not in self.choices:
                raise TypeError(f"Value is not in the choices of {self.choices}")

        if self.gt is not None:
            if value <= self.gt:
                raise TypeError(f"Value is not greater than {self.gt}")

        if self.ge is not None:
            if value < self.ge:
                raise TypeError(f"Value is not greater than or equal to {self.ge}")

        if self.lt is not None:
            if value >= self.lt:
                raise TypeError(f"Value is not less than {self.lt}")

        if self.le is not None:
            if value > self.le:
                raise TypeError(f"Value is not less than or equal to {self.le}")

        if self.multiple_of is not None:
            if value % self.multiple_of != 0:
                raise TypeError(f"Value is not a multiple of {self.multiple_of}")

        if self.allow_inf_nan is not None:
            if not self.allow_inf_nan and (
                value == float("inf") or value == float("-inf") or value == float("nan")
            ):
                raise TypeError("Value is not allowed to be infinity or nan")

        if self.max_digits is not None:
            if len(str(value)) > self.max_digits:
                raise TypeError(f"Value has more than {self.max_digits} digits")

        if self.decimal_places is not None:
            if len(str(value).split(".")[1]) > self.decimal_places:
                raise TypeError(
                    f"Value has more than {self.decimal_places} decimal places"
                )

        if self.min_length is not None:
            if len(str(value)) < self.min_length:
                raise TypeError(f"Value has less than {self.min_length} characters")

        if self.max_length is not None:
            if len(str(value)) > self.max_length:
                raise TypeError(f"Value has more than {self.max_length} characters")

    def to_io_schema(self) -> IOVariable:
        return IOVariable(
            run_io_type=(
                RunIOType.dictionary
                if (
                    inspect.isclass(self.type_class)
                    and issubclass(self.type_class, InputSchema)
                )
                else RunIOType.from_object(self.type_class)
            ),
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
            # Backwards compatible with older pipelines
            default=self.default if hasattr(self, "default") else None,
        )


class File(Variable):
    path: Path | None
    url: ParseResult | None
    remote_id: str | None

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        remote_id: str | None = None,
        url: str | None = None,
        title: str | None = None,
        allow_out_of_context_creation: bool = True,
        local_id: str | None = None,
    ) -> None:
        super().__init__(
            type_class=self.__class__,
            is_input=False,
            is_output=False,
            title=title,
            local_id=local_id,
            allow_out_of_context_creation=allow_out_of_context_creation,
        )

        self.path = (
            path
            if isinstance(path, Path)
            else (Path(path) if path is not None else None)
        )
        self.remote_id: str | None = remote_id
        self.url = urlparse(url) if url is not None else None

    def save(self, path: str | Path) -> None:
        if self.path is None and self.url is None:
            raise Exception("Path and URL are None")

        if isinstance(path, str):
            path = Path(path)

        if self.path is not None and self.path.is_dir():
            raise Exception("Path is a directory")

        if self.url is not None:
            with path.open("wb") as f:
                with httpx.stream("GET", self.url.geturl()) as response:
                    total = int(response.headers["Content-Length"])

                    with tqdm(
                        total=total, unit_scale=True, unit_divisor=1024, unit="B"
                    ) as progress:
                        num_bytes_downloaded = response.num_bytes_downloaded
                        for chunk in response.iter_bytes():
                            f.write(chunk)
                            progress.update(
                                response.num_bytes_downloaded - num_bytes_downloaded
                            )
                            num_bytes_downloaded = response.num_bytes_downloaded

        elif self.path is not None:
            path.write_bytes(self.path.read_bytes())


class Directory(File):
    def __init__(
        self,
        *,
        path: str | Path | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            path=path,
            **kwargs,
        )

        if (
            self.path is not None
            and not self.path.is_dir()
            and not str(self.path).endswith(".zip")
            and self.remote_id is None
        ):
            raise Exception("Path is not a directory")

    @classmethod
    def from_object(
        cls,
        *args,
        **kwargs,
    ):
        raise NotImplementedError("Directory.from_object is not implemented")


ST = TypeVar("ST")


class Stream(Generic[ST], Iterable):
    def __init__(self, iterable: Iterable[ST]):
        self.iterable = iterable

    def __iter__(self):
        return self.iterable.__iter__()

    def __next__(self):
        return self.iterable.__next__()


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
            # At the moment only the File variable can be used on startup
            if isinstance(var, File):
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
            raise RunInputException(
                "Mismatch of number of inputs, expecting %u got %s"
                % (len(input_variables), len(inputs))
            )

        self._startup()

        running_variables = {}

        # Add all File's to the running variables
        for var in self.variables:
            if isinstance(var, File):
                if not var.path:
                    raise RunInputException("Must define a path for a File")

                running_variables[var.local_id] = var

        for i, input in enumerate(inputs):
            target_type = input_variables[i].type_class

            if issubclass(target_type, InputSchema) and isinstance(input, dict):
                input = target_type(**input)
            elif not isinstance(input, input_variables[i].type_class):
                if isinstance(input, int) and input_variables[i].type_class == float:
                    input = float(input)
                else:
                    raise RunInputException(
                        "Input type mismatch, expected %s got %s"
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
