from __future__ import annotations

import io
import json
import os
import urllib.parse
from typing import TYPE_CHECKING, Any, List, Optional, Union

import requests
from requests_toolbelt.multipart import encoder
from tqdm import tqdm

from pipeline.schemas.file import FileGet
from pipeline.schemas.function import FunctionCreate, FunctionGet
from pipeline.schemas.model import ModelCreate, ModelGet
from pipeline.schemas.pipeline import PipelineCreate, PipelineGet, PipelineVariableGet
from pipeline.schemas.run import RunCreate
from pipeline.util import generate_id, python_object_to_hex, python_object_to_name
from pipeline.util.logging import PIPELINE_STR

if TYPE_CHECKING:
    from pipeline.objects import Function, Graph, Model


class PipelineCloud:
    token: Optional[str]
    url: Optional[str]

    def __init__(self, url: str = None) -> None:
        self.token = os.getenv("PIPELINE_API_TOKEN")
        self.url = url or os.getenv("PIPELINE_API_URL", "https://api.pipeline.ai")

    def authenticate(self, token: str = None):
        """
        Authenticate with the pipeline.ai API
        """
        print("Authenticating")
        _token = token or self.token
        if _token is None:
            raise Exception("Please pass a valid token or set it as an env var")
        status_url = urllib.parse.urljoin(self.url, "/v2/users/me")

        response = requests.get(
            status_url, headers={"Authorization": "Bearer %s" % _token}
        )

        response.raise_for_status()

        if response.json():
            print("Succesfully authenticated with the Pipeline API (%s)" % self.url)
        self.token = _token

    def upload_file(self, file_or_path, remote_path) -> FileGet:
        if isinstance(file_or_path, str):
            with open(file_or_path, "rb") as file:
                return self._post_file("/v2/files/", file, remote_path)
        else:
            return self._post_file("/v2/files/", file_or_path, remote_path)

    def upload_python_object_to_file(self, obj, remote_path) -> FileGet:
        return self.upload_file(
            io.BytesIO(python_object_to_hex(obj).encode()), remote_path
        )

    def _post(self, endpoint, json_data):

        headers = {
            "Authorization": "Bearer %s" % self.token,
            "Content-type": "application/json",
        }

        url = urllib.parse.urljoin(self.url, endpoint)
        response = requests.post(url, headers=headers, json=json_data)
        response.raise_for_status()
        return response.json()

    def _post_file(self, endpoint, file, remote_path):
        if not hasattr(file, "name"):
            file.name = generate_id(20)

        e = encoder.MultipartEncoder(
            fields={
                "file_path": remote_path,
                "file": (
                    file.name,
                    file,
                    "application/octet-stream",
                    {"Content-Transfer-Encoding": "binary"},
                ),
            }
        )
        encoder_len = e.len
        bar = tqdm(
            desc=f"{PIPELINE_STR} Uploading",
            unit="B",
            unit_scale=True,
            total=encoder_len,
            unit_divisor=1024,
        )

        def progress_callback(monitor):
            bar.n = monitor.bytes_read
            bar.refresh()
            if monitor.bytes_read == encoder_len:
                bar.close()

        encoded_stream_data = encoder.MultipartEncoderMonitor(
            e, callback=progress_callback
        )

        headers = {
            "Authorization": "Bearer %s" % self.token,
            "Content-type": encoded_stream_data.content_type,
        }
        url = urllib.parse.urljoin(self.url, endpoint)
        response = requests.post(url, headers=headers, data=encoded_stream_data)
        response.raise_for_status()
        return FileGet.parse_obj(response.json())

    def upload_function(self, function: Function) -> FunctionGet:
        inputs = [
            dict(name=name, type_name=python_object_to_name(type))
            for name, type in function.typing_inputs.items()
        ]
        output = [
            dict(name=name, type_name=python_object_to_name(type))
            for name, type in function.typing_outputs.items()
        ]

        file_schema = self.upload_python_object_to_file(function, "/lol")

        function_create_schema = FunctionCreate(
            local_id=function.local_id,
            name=function.name,
            function_source=function.source,
            hash=function.hash,
            inputs=inputs,
            output=output,
            file_id=file_schema.id,
        )

        request_result = self._post("/v2/functions/", function_create_schema.dict())

        return FunctionGet.parse_obj(request_result)

    def upload_model(self, model: Model) -> ModelGet:
        file_schema = self.upload_python_object_to_file(model, "/lol")

        model_create_schema = ModelCreate(
            local_id=model.local_id,
            name=model.name,
            model_source=model.source,
            hash=model.hash,
            file_id=file_schema.id,
        )

        request_result = self._post("/v2/models/", model_create_schema.dict())

        return ModelGet.parse_obj(request_result)

    def upload_pipeline(self, new_pipeline_graph: Graph) -> PipelineGet:

        new_name = new_pipeline_graph.name
        print("Uploading functions")
        new_functions = [
            self.upload_function(_function)
            for _function in new_pipeline_graph.functions
        ]

        print("Uploading models")
        new_models = [self.upload_model(_model) for _model in new_pipeline_graph.models]

        new_variables: List[PipelineVariableGet] = []
        print("Uploading variables")
        for _var in new_pipeline_graph.variables:
            _var_type_file = self.upload_file(
                io.BytesIO(python_object_to_hex(_var.type_class).encode()), "/"
            )
            _var_schema = PipelineVariableGet(
                local_id=_var.local_id,
                name=_var.name,
                type_file=_var_type_file,
                is_input=_var.is_input,
                is_output=_var.is_output,
            )
            new_variables.append(_var_schema)

        new_graph_nodes = [
            _node.to_create_schema() for _node in new_pipeline_graph.nodes
        ]
        new_outputs = [_output.local_id for _output in new_pipeline_graph.outputs]

        pipeline_create_schema = PipelineCreate(
            name=new_name,
            variables=new_variables,
            functions=new_functions,
            models=new_models,
            graph_nodes=new_graph_nodes,
            outputs=new_outputs,
        )
        print("Uploading pipeline graph")
        request_result = self._post(
            "/v2/pipelines", json.loads(pipeline_create_schema.json())
        )

        return PipelineGet.parse_obj(request_result)

    def run_pipeline(
        self,
        pipeline_id_or_schema: Union[str, PipelineGet],
        data_or_file_id: Union[Any, str],
    ):
        # TODO: Add support for generic object inference. Only strs at the moment.
        file_id = None
        if isinstance(data_or_file_id, str):
            file_id = data_or_file_id
        else:
            temp_file = io.BytesIO(python_object_to_hex(data_or_file_id).encode())
            uploaded_data = self.upload_file(temp_file, "/")
            file_id = uploaded_data.id
        print(file_id)

        pipeline_id = None
        if isinstance(pipeline_id_or_schema, str):
            pipeline_id = pipeline_id_or_schema
        elif isinstance(pipeline_id_or_schema, PipelineGet):
            pipeline_id = pipeline_id_or_schema.id
        else:
            raise Exception(
                (
                    "Must either pass a pipeline id, or a pipeline get schema. "
                    "Not object of type %s in arg 1." % str(pipeline_id_or_schema)
                )
            )

        # TODO: swap "data=data_or_file_id" for "file_id=file_id" later
        # when the generic object inference is added back.
        run_create_schema = RunCreate(pipeline_id=pipeline_id, data=data_or_file_id)

        return self._post("/v2/runs", json.loads(run_create_schema.json()))
