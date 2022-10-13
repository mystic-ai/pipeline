from __future__ import annotations

import hashlib
import io
import json
import os
import urllib.parse
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Type, Union

import httpx
import requests
from pydantic import ValidationError
from requests_toolbelt.multipart import encoder
from tqdm import tqdm

from pipeline.exceptions.InvalidSchema import InvalidSchema
from pipeline.exceptions.MissingActiveToken import MissingActiveToken
from pipeline.schemas.base import BaseModel
from pipeline.schemas.compute_requirements import ComputeRequirements
from pipeline.schemas.data import DataGet
from pipeline.schemas.file import FileCreate, FileGet
from pipeline.schemas.function import FunctionCreate, FunctionGet
from pipeline.schemas.model import ModelCreate, ModelGet
from pipeline.schemas.pipeline import (
    PipelineCreate,
    PipelineFileVariableGet,
    PipelineGet,
    PipelineVariableGet,
)
from pipeline.schemas.pipeline_file import (
    MultipartUploadMetadata,
    PipelineFileDirectUploadFinaliseCreate,
    PipelineFileDirectUploadInitCreate,
    PipelineFileDirectUploadInitGet,
    PipelineFileDirectUploadPartCreate,
    PipelineFileDirectUploadPartGet,
    PipelineFileGet,
)
from pipeline.schemas.run import RunCreate, RunGet
from pipeline.util import (
    generate_id,
    hex_to_python_object,
    python_object_to_hex,
    python_object_to_name,
)
from pipeline.util.logging import PIPELINE_FILE_STR, PIPELINE_STR

if TYPE_CHECKING:
    from pipeline.objects import Function, Graph, Model

FILE_CHUNK_SIZE = 200 * 1024 * 1024  # 200 MiB


class PipelineCloud:
    token: Optional[str]
    url: Optional[str]

    def __init__(
        self,
        *,
        url: str = None,
        token: str = None,
        timeout=60.0,
        verbose=True,
    ) -> None:
        self.token = token or os.getenv("PIPELINE_API_TOKEN")
        self.url = url or os.getenv("PIPELINE_API_URL", "https://api.pipeline.ai")
        self.timeout = timeout
        self.verbose = verbose
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
        if self.verbose:
            print("Authenticating")

        _token = token or self.token
        if _token is None:
            raise MissingActiveToken(
                token="", message="Please pass a valid token or set it as an ENV var"
            )
        status_url = urllib.parse.urljoin(self.url, "/v2/users/me")

        response = requests.get(
            status_url,
            headers={"Authorization": "Bearer %s" % _token},
            timeout=self.timeout,
        )

        if (
            response.status_code == HTTPStatus.FORBIDDEN
            or response.status_code == HTTPStatus.UNAUTHORIZED
        ):
            raise MissingActiveToken(token=_token)
        else:
            response.raise_for_status()

        if response.json() and self.verbose:
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

    @staticmethod
    def _get_raise_for_status(
        response: Union[requests.Response, httpx.Response]
    ) -> None:
        # A handler for errors that might be sent with messages from Top.
        if (isinstance(response, requests.Response) and not response.ok) or (
            isinstance(response, httpx.Response)
            and not response.status_code == httpx.codes.OK
        ):
            content = response.json()
            # Every exception has content of {detail, status_code[, headers]}
            # TODO Some exceptions in Top send detail as a string and not a dict.
            # These exceptions are now handled like normal HTTP exceptions.
            # Need to rewrite these to all have the same format.
            detail = content.pop("detail", "")
            message = None
            # In most cases detail is not a string but a dict.
            if isinstance(detail, dict):
                message = detail.pop("message", None)
            elif not isinstance(detail, str):
                detail = ""
            detail = detail or ""
            # If a message was delivered we want to show that. Otherwise it's a
            # standard HTTP error and should be handled by raise_for_status()
            if message is not None:
                raise Exception(f"Error {response.status_code}: {message} {detail}")
            else:
                response.raise_for_status()

    def upload_file(self, file_or_path, remote_path) -> FileGet:

        if isinstance(file_or_path, str):
            # TODO: Change this to wrap the file object reader to convert to hex
            # everytime anything is read instead of reading it all at once.

            with open(file_or_path, "rb") as file:
                buffer = file.read()
            hex_buffer = buffer.hex()
            return self._post_file(
                "/v2/files/", io.BytesIO(hex_buffer.encode()), remote_path
            )
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

    def _initialise_direct_pipeline_file_upload(self, file_size: int) -> str:
        """Initialise a direct multi-part pipeline file upload"""
        direct_upload_schema = PipelineFileDirectUploadInitCreate(file_size=file_size)
        response = self._post(
            "/v2/pipeline-files/initiate-multipart-upload", direct_upload_schema.dict()
        )
        direct_upload_get = PipelineFileDirectUploadInitGet.parse_obj(response)
        return direct_upload_get.pipeline_file_id

    def _direct_upload_pipeline_file_chunk(
        self, data: bytes, pipeline_file_id: str, part_num: int
    ) -> MultipartUploadMetadata:
        """Upload a single chunk of a multi-part pipeline file upload.

        Returns the metadata associated with this upload (this is needed to pass into
        the finalisation step).
        """
        # get presigned URL
        part_upload_schema = PipelineFileDirectUploadPartCreate(
            pipeline_file_id=pipeline_file_id, part_num=part_num
        )
        response = self._post(
            "/v2/pipeline-files/presigned-url", part_upload_schema.dict()
        )
        part_upload_get = PipelineFileDirectUploadPartGet.parse_obj(response)
        # upload file chunk
        # convert data to hex
        data = data.hex().encode()
        response = requests.put(
            part_upload_get.upload_url, data=data, timeout=self.timeout
        )
        etag = response.headers["ETag"]
        return MultipartUploadMetadata(ETag=etag, PartNumber=part_num)

    def _finalise_direct_pipeline_file_upload(
        self, pipeline_file_id: str, multipart_metadata: List[MultipartUploadMetadata]
    ) -> PipelineFileGet:
        """Finalise the direct multi-part pipeline file upload"""
        finalise_upload_schema = PipelineFileDirectUploadFinaliseCreate(
            pipeline_file_id=pipeline_file_id,
            multipart_metadata=multipart_metadata,
        )
        response = self._post(
            "/v2/pipeline-files/finalise-multipart-upload",
            finalise_upload_schema.dict(),
        )
        return PipelineFileGet.parse_obj(response)

    def upload_pipeline_file(self, pipeline_file) -> PipelineFileVariableGet:
        """Upload PipelineFile given by pipeline_file.

        Since PipelineFiles can be very large, we implement this slightly
        differently to regular file uploads:
        - We need to split the file into chunks based on FILE_CHUNK_SIZE
        - We first initialise the multi-part upload with the server
        - We then upload the file in chunks (requesting a presigned upload URL for each
            chunk beforehand)
        - Lastly, we finalise the multi-part upload with the server
        """

        file_hash = self._hash_file(pipeline_file.path)
        file_size = os.path.getsize(pipeline_file.path)

        pipeline_file_id = self._initialise_direct_pipeline_file_upload(
            file_size=file_size
        )

        parts = []
        if self.verbose:
            progress = tqdm(
                desc=f"{PIPELINE_FILE_STR} Uploading {pipeline_file.path}",
                unit="B",
                unit_scale=True,
                total=file_size,
                unit_divisor=1024,
            )
        with open(pipeline_file.path, "rb") as f:
            while True:
                file_data = f.read(FILE_CHUNK_SIZE)
                if not file_data:
                    if self.verbose:
                        progress.close()
                    break

                part_num = len(parts) + 1

                upload_metadata = self._direct_upload_pipeline_file_chunk(
                    data=file_data,
                    pipeline_file_id=pipeline_file_id,
                    part_num=part_num,
                )
                parts.append(upload_metadata)
                if self.verbose:
                    progress.update(len(file_data))

        pipeline_file_get = self._finalise_direct_pipeline_file_upload(
            pipeline_file_id=pipeline_file_id, multipart_metadata=parts
        )
        return PipelineFileVariableGet(
            path=pipeline_file.path, file=pipeline_file_get.hex_file, hash=file_hash
        )

    def _get(self, endpoint: str, params: Dict[str, Any] = None):
        headers = {
            "Authorization": "Bearer %s" % self.token,
        }

        url = urllib.parse.urljoin(self.url, endpoint)
        response = requests.get(
            url, headers=headers, params=params, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def _post(self, endpoint, json_data):
        self.raise_for_invalid_token()
        headers = {
            "Authorization": "Bearer %s" % self.token,
            "Content-type": "application/json",
        }

        url = urllib.parse.urljoin(self.url, endpoint)
        response = requests.post(
            url, headers=headers, json=json_data, timeout=self.timeout
        )

        if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
            schema = json_data
            raise InvalidSchema(schema=schema)
        else:
            self._get_raise_for_status(response)

        return response.json()

    def _post_file(self, endpoint, file, remote_path) -> FileGet:
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
        if self.verbose:
            bar = tqdm(
                desc=f"{PIPELINE_STR} Uploading",
                unit="B",
                unit_scale=True,
                total=encoder_len,
                unit_divisor=1024,
            )
        if self.verbose:

            def progress_callback(monitor):
                bar.n = monitor.bytes_read
                bar.refresh()
                if monitor.bytes_read == encoder_len:
                    bar.close()

        encoded_stream_data = encoder.MultipartEncoderMonitor(
            e, callback=progress_callback if self.verbose else None
        )

        headers = {
            "Authorization": "Bearer %s" % self.token,
            "Content-type": encoded_stream_data.content_type,
        }
        url = urllib.parse.urljoin(self.url, endpoint)
        response = requests.post(
            url, headers=headers, data=encoded_stream_data, timeout=self.timeout
        )
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
        if new_pipeline_graph._has_run_startup:
            raise Exception("Cannot upload a pipeline that has already been run.")

        new_name = new_pipeline_graph.name
        if self.verbose:
            print("Uploading functions")
        new_functions = [
            self.upload_function(_function)
            for _function in new_pipeline_graph.functions
        ]
        for i, uploaded_function_schema in enumerate(new_functions):
            new_pipeline_graph.functions[i].local_id = uploaded_function_schema.id
        if self.verbose:
            print("Uploading models")
        new_models = [self.upload_model(_model) for _model in new_pipeline_graph.models]

        new_variables: List[PipelineVariableGet] = []
        if self.verbose:
            print("Uploading variables")

        from pipeline.objects import PipelineFile

        for _var in new_pipeline_graph.variables:
            _var_type_file = self.upload_file(
                io.BytesIO(python_object_to_hex(_var.type_class).encode()), "/"
            )

            pipeline_file_schema = None
            if isinstance(_var, PipelineFile):
                pipeline_file_schema = self.upload_pipeline_file(_var)

            _var_schema = PipelineVariableGet(
                local_id=_var.local_id,
                name=_var.name,
                type_file=_var_type_file,
                is_input=_var.is_input,
                is_output=_var.is_output,
                pipeline_file_variable=pipeline_file_schema,
            )

            new_variables.append(_var_schema)

        new_graph_nodes = [
            _node.to_create_schema() for _node in new_pipeline_graph.nodes
        ]
        new_outputs = [_output.local_id for _output in new_pipeline_graph.outputs]

        compute_requirements = None
        if new_pipeline_graph.min_gpu_vram_mb:
            compute_requirements = ComputeRequirements(
                min_gpu_vram_mb=new_pipeline_graph.min_gpu_vram_mb
            )

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
                compute_type=new_pipeline_graph.compute_type,
                compute_requirements=compute_requirements,
            )
        except ValidationError as e:
            raise InvalidSchema(schema="Graph", message=str(e))

        if self.verbose:
            print("Uploading pipeline graph")
        response = self._post(
            "/v2/pipelines", json.loads(pipeline_create_schema.json())
        )
        return PipelineGet.parse_obj(response)

    def run_pipeline(
        self,
        pipeline_id_or_schema: Union[str, PipelineGet],
        raw_data_or_schema: Union[Any, DataGet],
        compute_type: Optional[str] = None,
        min_gpu_vram_mb: Optional[int] = None,
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
                    compute_type (Optional[str]):
                        Compute type requirement for this run (e.g. "cpu" or "gpu"). If
                        not defined, it will fallback to that specified by the Pipeline
                        itself (typically this is "gpu")
                    min_gpu_vram_mb (Optional[int]):
                        Minimum amount of GPU VRAM required. If not defined, it will
                        fallback to that specified by the Pipeline (if applicable). This
                        should only be specified for GPU workloads.

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

        compute_requirements = None
        if min_gpu_vram_mb:
            compute_requirements = ComputeRequirements(min_gpu_vram_mb=min_gpu_vram_mb)

        run_create_schema = RunCreate(
            pipeline_id=pipeline_id,
            data_id=_data_id,
            compute_type=compute_type,
            compute_requirements=compute_requirements,
        )
        run_json: dict = self._post("/v2/runs", json.loads(run_create_schema.json()))
        return RunGet.parse_obj(run_json)

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

    def download_result(self, result_id_or_schema: Union[str, RunGet]) -> Any:
        """
        Downloads Result object from Pipeline Cloud.
            Parameters:
                    result_id_or_schema (Union[str, RunGet]):
                    The id for the desired run result
                    or the schema obtained from the run

            Returns:
                    object (Any): De-Serialized run result file object.
        """
        result_id = None
        if isinstance(result_id_or_schema, str):
            result_id = result_id_or_schema
        else:
            try:
                result_id = RunGet.parse_obj(result_id_or_schema).result.id
            except ValidationError:
                raise InvalidSchema(
                    schema=result_id_or_schema,
                    message=(
                        "Must either pass a result id, or a run get schema. "
                        "Not object of type %s in arg 1." % str(result_id_or_schema)
                    ),
                )
        endpoint = f"/v2/files/{result_id}"
        f_get_schema: FileGet = self._download_schema(
            schema=FileGet, endpoint=endpoint, params=dict(return_data=True)
        )
        return hex_to_python_object(f_get_schema.data)

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

    def _hash_file(self, file_path: str, block_size=2**20) -> str:
        md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            while True:
                data = f.read(block_size)
                if not data:
                    break
                md5.update(data)
        return md5.hexdigest()
