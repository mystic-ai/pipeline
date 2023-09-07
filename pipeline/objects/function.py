import inspect
import uuid
from hashlib import sha256
from typing import Any, Callable, Dict, Optional

from pipeline.util import generate_id


class Function:
    local_id: str
    remote_id: str

    name: str
    source: str
    hash: str

    typing_inputs: Dict[str, Any]
    typing_outputs: Dict[str, Any]

    function: Callable

    class_instance: Optional[Any]

    def __init__(
        self, function: Callable, *, remote_id: str = None, class_instance: Any = None
    ):
        self.name = function.__name__
        self.remote_id = remote_id
        self.class_instance = class_instance
        self.function = function

        try:
            self.source = inspect.getsource(function)
        except OSError:
            self.source = str(uuid.uuid4())
        self.hash = sha256(self.source.encode()).hexdigest()

        # TODO: Add verification that all inputs to function have a typing annotation,
        # except for "self"
        if "return" not in function.__annotations__:
            function.__annotations__["return"] = None

        self.typing_outputs = {"return": function.__annotations__["return"]}

        self.typing_inputs = {
            function_i: function.__annotations__[function_i]
            for function_i in function.__annotations__
            if not function_i == "return"
        }

        self.local_id = generate_id(10)
