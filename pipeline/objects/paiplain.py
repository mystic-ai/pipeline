import functools
from typing import Any, List, Tuple

from pipeline.objects.function import Function
from pipeline.objects.model import Model

_Results = List[Tuple[str, Any]]

# TODO figure out how to set input freely, like current Pipeline does
class Paiplain:
    stages: List[Function]
    stages_results: _Results
    pipeline_context_name: str = None
    models: List[Model]

    def __init__(self, new_pipeline_name):
        self.pipeline_context_name = new_pipeline_name
        self.stages = []
        self.stages_results = []
        self.models = []

    def get_results(self) -> List[Any]:
        return [res[1] for res in self.stages_results]

    def get_named_results(self) -> _Results:
        return self.stages_results

    def process(self, *data) -> _Results:
        for stage in self.stages:
            new_data = self._run_stage(stage, *data)
            if new_data is None:
                break
            data = new_data
        return self.stages_results

    def _run_stage(self, stage: Function, *data) -> List[Any]:
        try:
            res = stage.function(*data)
            self.stages_results.append((stage.function.__name__, res))
            if isinstance(res, List):
                return res
            return [res]
        except Exception as e:
            print(e)
            # TODO decide what to do with exception
            return None

    def stage(self, function):  # rename to "function" to preserve interface
        @functools.wraps(function)
        def wrap(*args, **kwargs):
            print("add step", function.__name__)
            function_ios = function.__annotations__
            if "return" not in function_ios:
                raise Exception(
                    "Must include an output type e.g.",
                    "'def my_func(...) -> int:'",
                )
            self.stages.append(Function(function))

        return wrap()

    def set_stages(self, *stages):
        self.stages = [Function(stage) for stage in stages]

    def model(
        self, model_class=None, *, file_or_dir: str = None, compress_tar=False
    ):
        pass
