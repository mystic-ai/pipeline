import io
import json
import typing as t
from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class RunState(int, Enum):
    created: int = 0
    routing: int = 1
    resource_accepted: int = 2
    # this includes creating a virtual environment and installing
    # packages
    creating_environment: int = 3
    # starting subrprocess running worker in custom environment
    starting_worker: int = 4
    downloading_graph: int = 5
    caching_graph: int = 6
    running: int = 7
    resource_returning: int = 8
    api_received: int = 9
    celery_worker_received: int = 22

    in_queue: int = 16
    denied: int = 11
    resource_rejected: int = 14
    resource_died: int = 15
    retrying: int = 13
    rerouting: int = 21

    completed: int = 10
    failed: int = 12
    rate_limited: int = 17
    lost: int = 18
    no_environment_installed: int = 19

    unknown: int = 20

    @classmethod
    def __get_validators__(cls):
        cls.lookup = {v: k.value for v, k in cls.__members__.items()}
        cls.value_lookup = {k.value: v for v, k in cls.__members__.items()}
        yield cls.validate

    @classmethod
    def validate(cls, v):
        try:
            v = int(v)
        except Exception:
            ...

        if isinstance(v, str):
            return cls.lookup[v]
        elif isinstance(v, int):
            return getattr(cls, cls.value_lookup[v])
        else:
            raise ValueError(f"Invalid value: {v}")


class RunError(Enum):
    input_error = 1
    unroutable = 2
    graph_error = 3
    runtime_error = 4


class RunFileType(Enum):
    input = "input"
    output = "output"


class RunFile(BaseModel):
    id: str
    run_id: str
    io_type: RunFileType
    path: str

    class Config:
        # use_enum_values = True
        orm_mode = True


class RunIOType(str, Enum):
    integer: str = "integer"
    string: str = "string"
    fp: str = "fp"
    dictionary: str = "dictionary"
    boolean: str = "boolean"
    none: str = "none"
    array: str = "array"

    pkl: str = "pkl"
    file: str = "file"

    @classmethod
    def from_object(cls, obj: t.Any):
        # Get the enum type for the object.
        if isinstance(obj, int):
            return cls.integer
        elif isinstance(obj, float):
            return cls.fp
        elif isinstance(obj, str):
            return cls.string
        elif isinstance(obj, bool):
            return cls.boolean
        elif obj is None:
            return cls.none
        elif isinstance(obj, dict):
            try:
                json.dumps(obj)
            except (TypeError, OverflowError):
                return cls.pkl
            return cls.dictionary
        elif isinstance(obj, list):
            try:
                json.dumps(obj)
            except (TypeError, OverflowError):
                return cls.pkl
            return cls.array
        elif isinstance(obj, io.BufferedIOBase):
            return cls.file
        else:
            return cls.pkl


class RunOutputFile(BaseModel):
    name: str
    path: str
    url: str
    size: int


class RunOutput(BaseModel):
    type: RunIOType
    value: t.Optional[t.Any]
    file: t.Optional[RunOutputFile]


class RunResult(BaseModel):
    run_id: str
    outputs: t.List[RunOutput]

    def result_array(self) -> t.List[t.Any]:
        return [output.value for output in self.outputs]


class Run(BaseModel):
    id: str

    created_at: datetime

    pipeline_id: str
    environment_id: str
    environment_hash: str

    state: RunState

    error: t.Optional[t.Tuple[RunError, str]]

    result: t.Optional[RunResult]

    class Config:
        # use_enum_values = True
        orm_mode = True


class RunStateTransition(BaseModel):
    run_id: str
    new_state: RunState
    time: datetime


class RunInput(BaseModel):
    type: RunIOType
    value: t.Any

    file_name: t.Optional[str]
    file_path: t.Optional[str]


class RunCreate(BaseModel):
    pipeline_id_or_pointer: str
    input_data: t.List[RunInput]
    async_run: bool = False
