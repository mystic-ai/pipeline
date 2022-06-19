from __future__ import annotations

import io
import json
import os
import urllib.parse
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Type, Union

import requests
from pydantic import ValidationError
from requests_toolbelt.multipart import encoder
from tqdm import tqdm

from pipeline.exceptions.InvalidSchema import InvalidSchema
from pipeline.exceptions.MissingActiveToken import MissingActiveToken
from pipeline.schemas.base import BaseModel
from pipeline.schemas.data import DataGet
from pipeline.schemas.file import FileCreate, FileGet
from pipeline.schemas.function import FunctionCreate, FunctionGet
from pipeline.schemas.model import ModelCreate, ModelGet
from pipeline.schemas.pipeline import PipelineCreate, PipelineGet, PipelineVariableGet
from pipeline.schemas.run import RunCreate
from pipeline.util import (
    generate_id,
    hex_to_python_object,
    python_object_to_hex,
    python_object_to_name,
)
from pipeline.util.logging import PIPELINE_STR

if TYPE_CHECKING:
    from pipeline.objects import Function, Graph, Model


class PipelineCloud:
    token: Optional[str]
    url: Optional[str]

    def __init__(self, url: str = None, token: str = None) -> None:
        self.token = token or os.getenv("PIPELINE_API_TOKEN")
        self.url = url or os.getenv("PIPELINE_API_URL", "https://api.pipeline.ai")
        self.__valid_token__ = False
        if self.token is not None:
            self.authenticate()

    def authenticate(self, token: str = None):
        """
        Authenticate with the pipeline.ai API
            Parameters:
                token (str): API user token for authentication.
                    Pass it as an arg or set it as an ENV var.
            Returns:
                None
        """
        print("Authenticating")
        _token = token or self.token
        if _token is None:
            raise MissingActiveToken(
                token="", message="Please pass a valid token or set it as an ENV var"
            )
        status_url = urllib.parse.urljoin(self.url, "/v2/users/me")

        response = requests.get(
            status_url, headers={"Authorization": "Bearer %s" % _token}
        )

        if (
            response.status_code == HTTPStatus.FORBIDDEN
            or response.status_code == HTTPStatus.UNAUTHORIZED
        ):
            raise MissingActiveToken(token=_token)
        else:
            response.raise_for_status()

        if response.json():
            print("Succesfully authenticated with the Pipeline API (%s)" % self.url)
        self.token = _token
        self.__valid_token__ = True

    def raise_for_invalid_token(self):
        if not self.__valid_token__:
            raise MissingActiveToken(
                token=self.token,
                message=(
                    "Please set a valid token as an ENV var "
                    "and call authenticate(); "
                    "or pass a valid token as a parameter authenticate(token)"
                ),
            )

    def upload_file(self, file_or_path, remote_path) -> FileGet:

        if isinstance(file_or_path, str):
            with open(file_or_path, "rb") as file:
                return self._post_file("/v2/files/", file, remote_path)
        else:
            return self._post_file("/v2/files/", file_or_path, remote_path)

    def upload_data(self, file_or_path, remote_path) -> DataGet:
        uploaded_file = self.upload_file(file_or_path, remote_path)
        uploaded_data = self._post("/v2/data", uploaded_file.dict())
        return DataGet.parse_obj(uploaded_data)

    def upload_python_object_to_file(self, obj, remote_path) -> FileGet:
        return self.upload_file(
            io.BytesIO(python_object_to_hex(obj).encode()), remote_path
        )

    def _get(self, endpoint: str, params: Dict[str, Any] = None):
        headers = {
            "Authorization": "Bearer %s" % self.token,
        }

        url = urllib.parse.urljoin(self.url, endpoint)
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    def _post(self, endpoint, json_data):
        self.raise_for_invalid_token()
        headers = {
            "Authorization": "Bearer %s" % self.token,
            "Content-type": "application/json",
        }

        url = urllib.parse.urljoin(self.url, endpoint)
        response = requests.post(url, headers=headers, json=json_data)

        if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
            schema = json_data
            raise InvalidSchema(schema=schema)
        else:
            response.raise_for_status()

        return response.json()

    def _post_file(self, endpoint, file, remote_path):
        self.raise_for_invalid_token()
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
        if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
            schema = FileCreate.__name__
            raise InvalidSchema(schema=schema)
        else:
            response.raise_for_status()
        return FileGet.parse_obj(response.json())

    def upload_function(self, function: Function) -> FunctionGet:
        try:
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
        except AttributeError as e:
            raise InvalidSchema(schema="Function", message=str(e))
        response = self._post("/v2/functions", function_create_schema.dict())
        return FunctionGet.parse_obj(response)

    def upload_model(self, model: Model) -> ModelGet:
        file_schema = self.upload_python_object_to_file(model, "/lol")
        try:
            model_create_schema = ModelCreate(
                local_id=model.local_id,
                name=model.name,
                model_source=model.source,
                hash=model.hash,
                file_id=file_schema.id,
            )
        except ValidationError as e:
            raise InvalidSchema(schema="Model", message=str(e))

        response = self._post("/v2/models", model_create_schema.dict())
        return ModelGet.parse_obj(response)

    def upload_pipeline(
        self,
        new_pipeline_graph: Graph,
        public: bool = False,
        description: str = "",
        tags: Set[str] = None,
    ) -> PipelineGet:
        """
        Upload a Pipeline to the Cloud.

            Parameters:
                    new_pipeline_graph (Graph): Graph repr. Pipeline to be uploaded.
                        Obtained from Pipeline.get_pipeline(name:str)
                    public (bool): If pipeline should be visible to public.
                        Defaults to False.
                    description (str): Description of the Pipeline.
                    tags (Set[str]): Set of tags for the pipeline. Eg: {"BERT", "NLP"}

            Returns:
                    pipeline (PipelineGet): Object representing uploaded pipeline.
        """

        new_name = new_pipeline_graph.name
        print("Uploading functions")
        new_functions = [
            self.upload_function(_function)
            for _function in new_pipeline_graph.functions
        ]
        for i, uploaded_function_schema in enumerate(new_functions):
            new_pipeline_graph.functions[i].local_id = uploaded_function_schema.id

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
        try:
            pipeline_create_schema = PipelineCreate(
                name=new_name,
                variables=new_variables,
                functions=new_functions,
                models=new_models,
                graph_nodes=new_graph_nodes,
                outputs=new_outputs,
                public=public,
                description=description,
                tags=tags or set(),
            )
        except ValidationError as e:
            raise InvalidSchema(schema="Graph", message=str(e))

        print("Uploading pipeline graph")
        response = self._post(
            "/v2/pipelines", json.loads(pipeline_create_schema.json())
        )
        return PipelineGet.parse_obj(response)

    def run_pipeline(
        self,
        pipeline_id_or_schema: Union[str, PipelineGet],
        raw_data_or_schema: Union[Any, DataGet],
    ):
        """
        Uploads Data and executes a Run of given pipeline over given data.

            Parameters:
                    pipeline_id_or_schema (Union[str, PipelineGet]):
                        The id for the desired pipeline
                        or the schema obtained from uploading it
                    raw_data_or_schema (Union[Any, DataGet]):
                        Raw data for Pipeline execution
                        or schema for already uploaded data.

            Returns:
                    run (Any): Run object containing metadata and outputs.
        """
        # TODO: Add support for generic object inference. Only strs at the moment.
        if not isinstance(raw_data_or_schema, DataGet):
            temp_file = io.BytesIO(python_object_to_hex(raw_data_or_schema).encode())
            uploaded_data = self.upload_data(temp_file, "/")
            _data_id = uploaded_data.id
        elif isinstance(raw_data_or_schema, DataGet):
            _data_id = raw_data_or_schema.id
        else:
            raise Exception("Must either pass a raw data, or DataGet schema.")

        pipeline_id = None
        if isinstance(pipeline_id_or_schema, str):
            pipeline_id = pipeline_id_or_schema
        elif isinstance(pipeline_id_or_schema, PipelineGet):
            pipeline_id = pipeline_id_or_schema.id
        else:
            raise InvalidSchema(
                schema=pipeline_id_or_schema,
                message=(
                    "Must either pass a pipeline id, or a pipeline get schema. "
                    "Not object of type %s in arg 1." % str(pipeline_id_or_schema)
                ),
            )

        run_create_schema = RunCreate(pipeline_id=pipeline_id, data_id=_data_id)
        return self._post("/v2/runs", json.loads(run_create_schema.json()))

    def _download_schema(
        self, schema: Type[BaseModel], endpoint: str, params: Optional[Dict[str, Any]]
    ) -> Type[BaseModel]:
        """
        Request json data to a given endpoint and parse it into given schema

            Parameters:
                schema (Type[BaseModel]): Which schema is expected to be returned
                endpoint (str): endpoint to which the request must be sent
                params (Optional[Dict[str, Any]]): optional request params

            Returns:
                schema (Type[BaseModel]): The populated schema passed in
        """
        response = self._get(endpoint=endpoint, params=params)
        try:
            return schema(**response)
        except ValidationError as e:
            raise InvalidSchema(schema=schema.__name__, message=str(e))

    def download_function(self, id: str) -> Function:
        """
        Downloads Function object from Pipeline Cloud.

            Parameters:
                    id (str):
                        The id for the desired function

            Returns:
                    function (Function): De-Serialized function.
        """
        endpoint = f"/v2/functions/{id}"
        f_get_schema: FunctionGet = self._download_schema(
            schema=FunctionGet,
            endpoint=endpoint,
            params=dict(return_data=True),
        )
        # FIXME we have a circular import issue that needs reviewing
        from pipeline.objects import Function

        return Function.from_schema(f_get_schema)

    def download_model(self, id: str) -> Model:
        """
        Downloads Model object from Pipeline Cloud.

            Parameters:
                    id (str):
                        The id for the desired model

            Returns:
                    model (Model): De-Serialized model.
        """
        endpoint = f"/v2/models/{id}"
        m_get_schema: ModelGet = self._download_schema(
            schema=ModelGet,
            endpoint=endpoint,
            params=dict(return_data=True),
        )
        # FIXME we have a circular import issue that needs reviewing
        from pipeline.objects import Model

        return Model.from_schema(m_get_schema)

    def download_data(self, id: str) -> Any:
        """
        Downloads Data object from Pipeline Cloud.

            Parameters:
                    id (str):
                        The id for the desired data

            Returns:
                    object (Any): De-Serialized data object.
        """
        endpoint = f"/v2/data/{id}"
        d_get_schema: DataGet = self._download_schema(
            schema=DataGet,
            endpoint=endpoint,
            params=dict(return_data=True),
        )
        return hex_to_python_object(d_get_schema.hex_file.data)

    def download_pipeline(self, id: str) -> Graph:
        """
        Downloads Graph object from Pipeline Cloud.

            Parameters:
                    id (str):
                        The id for the desired pipeline

            Returns:
                    graph (Graph): De-Serialized pipeline.
        """
        endpoint = f"/v2/pipelines/{id}"
        p_get_schema: PipelineGet = self._download_schema(
            schema=PipelineGet,
            endpoint=endpoint,
            params=dict(return_data=True),
        )
        # FIXME we have a circular import issue that needs reviewing
        from pipeline.objects import Graph

        return Graph.from_schema(p_get_schema)
