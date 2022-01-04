import csv
import functools
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pipeline.api import authenticate
from pipeline.api.call import post
from pipeline.api.function import upload_function
from pipeline.api.run import run_pipeline
from pipeline.objects.function import Function
from pipeline.objects.model import Model
from pipeline.schemas.pipeline import (  # PipelineVariableGet,
    PipelineCreate,
    PipelineGet,
)
from pipeline.util.logging import _print

Results = List[Tuple[str, Any]]


# TODO figure out how to set input freely, like current Pipeline does
class PipelineV2:
    stages: List[Function]
    stages_results: Results
    pipeline_context_name: Optional[str] = None
    models: Dict[str, Model]
    remote_id: Optional[Union[str, PipelineGet]]

    def __init__(self, new_pipeline_name: str):
        self.pipeline_context_name = new_pipeline_name
        self.stages = []
        self.stages_results = []
        self.models = {}
        self.remote_id = None

    def get_results(self) -> List[Any]:
        return [res[1] for res in self.stages_results]

    def get_named_results(self) -> Results:
        return self.stages_results

    def run(self, *data: Any) -> Results:
        for stage in self.stages:
            new_data = self._run_stage(stage, *data)
            if new_data is None:
                break
            data = new_data
        return self.stages_results

    def _run_stage(self, stage: Function, *data: Any) -> Optional[Tuple[Any]]:
        try:
            # TODO change models to dict or tuple list or something
            qualname = stage.function.__qualname__.split(".")[-2]
            print(qualname)
            if qualname in self.models.keys():
                fles = self.models[qualname]
                res = stage.function(fles, *data)
            else:
                res = stage.function(*data)
            self.stages_results.append((stage.function.__name__, res))
            if isinstance(res, tuple):
                return res
            return (res,)
        except Exception as e:
            print(e)
            # TODO decide what to do with exception
            return None

    # rename to "function" to preserve interface
    def stage(self, function: Callable[[Any], Any]) -> None:
        @functools.wraps(function)
        def wrap(*args: Any, **kwargs: Any) -> None:
            print("add step", function.__name__)
            function_ios = function.__annotations__
            if "return" not in function_ios:
                raise Exception(
                    "Must include an output type e.g.",
                    "'def my_func(...) -> int:'",
                )
            self.stages.append(Function(function))

        return wrap()

    def set_stages(self, *stages: Callable[[Any], Any]) -> None:
        self.stages = [Function(stage) for stage in stages]

    def model(self, model: Any) -> None:
        # TODO run_model function?
        pipeline_model = Model(model)
        model_name: str = model.__class__.__qualname__.split(".")[-1]
        self.models[model_name] = pipeline_model

    def upload(self) -> PipelineGet:
        new_name = self.pipeline_context_name
        _print("Uploading stages")
        new_functions = [upload_function(_function) for _function in self.stages]

        # new_variables: List[PipelineVariableGet] = []
        # _print("Uploading variables")
        # for _var in new_pipeline_graph.variables:
        #     _var_type_file = upload_file(
        #         io.BytesIO(python_object_to_hex(_var.type_class).encode()),
        #         "/"
        #     )
        #     _var_schema = PipelineVariableGet(
        #         local_id=_var.local_id,
        #         name=_var.name,
        #         type_file=_var_type_file,
        #         is_input=_var.is_input,
        #         is_output=_var.is_output,
        #     )
        #     new_variables.append(_var_schema)

        # new_graph_nodes = [
        #     _node.to_create_schema() for _node in new_pipeline_graph.nodes
        # ]
        # new_outputs = [
        # _output.local_id for _output in new_pipeline_graph.outputs
        # ]
        # TODO tell remote -- Top, to set functions as stages
        # change schema perhaps
        pipeline_create_schema = PipelineCreate(
            name=new_name,
            # variables=new_variables,
            functions=new_functions,
            # graph_nodes=new_graph_nodes,
            # outputs=new_outputs,
        )
        _print("Uploading pipeline")
        request_result = post(
            "/v2/pipelines", json.loads(pipeline_create_schema.json())
        )
        pipe_get = PipelineGet.parse_obj(request_result)
        self.remote_id = pipe_get
        return pipe_get

    def auth(self) -> None:
        api_token = os.getenv("TOKEN")
        if api_token is None:
            raise EnvironmentError("missing TOKEN env var")
        authenticate(api_token)

    def run_remote(self, *initial_data: Any) -> Any:
        if self.remote_id is None:
            raise RuntimeError("undefined remote id, upload pipeline before run")
        return run_pipeline(self.remote_id, *initial_data)

    def save(
        self,
        relative_path: str = ".",
        filename: str = "pipeline",
        extension: str = "csv",
    ) -> None:
        f_name = os.path.join(relative_path, filename + "." + extension)
        with open(f_name, "w") as out:
            csv_out = csv.writer(out)
            csv_out.writerow(["Stage", "Results"])
            for row in self.stages_results:
                csv_out.writerow(row)
