from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import uuid
from http import HTTPStatus
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import dill
import httpx
from pydantic import ValidationError
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper

from pipeline import configuration
from pipeline.api.environments import PipelineCloudEnvironment, resolve_environment_id
from pipeline.exceptions.InvalidSchema import InvalidSchema
from pipeline.exceptions.MissingActiveToken import MissingActiveToken
from pipeline.objects.variable import PipelineFile
from pipeline.schemas.base import BaseModel
from pipeline.schemas.compute_requirements import ComputeRequirements
from pipeline.schemas.data import DataGet
from pipeline.schemas.file import FileFormat, FileGet
from pipeline.schemas.function import FunctionCreate, FunctionGet
from pipeline.schemas.model import ModelCreate, ModelGet
from pipeline.schemas.pipeline import (
    PipelineCreate,
    PipelineFileVariableGet,
    PipelineGet,
    PipelineVariableCreate,
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
    CallbackBytesIO,
    dump_object,
    load_object,
    package_version,
    python_object_to_name,
)
from pipeline.util.logging import PIPELINE_FILE_STR, PIPELINE_STR

if TYPE_CHECKING:
    from pipeline.objects import Function, Graph, Model

FILE_CHUNK_SIZE = 200 * 1024 * 1024  # 200 MiB
BINARY_MIME_TYPE = "application/octet-stream"
EnvironmentObjectOrID = Union[PipelineCloudEnvironment, str]


def _as_upload_file(object, name: Optional[str] = None) -> Tuple[str, BinaryIO, str]:
    """Represent `object` as an HTTP file upload.

    Returns a structure suitable for passing to the `files` parameter of
    httpx.request.
    """
    if name is None:
        name = str(uuid.uuid4())
    return (name, _as_binary_file(object), BINARY_MIME_TYPE)


def _as_binary_file(obj) -> BinaryIO:
    """Return `obj` as a pickle-encoded file opened in binary-read mode."""
    return io.BytesIO(dump_object(obj))


class PipelineCloud:
    token: Optional[str]
    url: Optional[str]

    def __init__(
        self,
        *,
        url: str = None,
        token: str = None,
        timeout: float = 60.0,
        verbose: bool = True,
    ):
        if url is None:
            url = os.environ.get(
                "PIPELINE_API_URL",
                configuration.DEFAULT_REMOTE,
            )
        if token is None:
            token = os.environ.get(
                "PIPELINE_API_TOKEN",
                configuration.remote_auth.get(url),
            )
        self._initialise_client(url, token, timeout)

        self.verbose = verbose
        self.__valid_token__ = False
        if self.token is not None:
            self.authenticate()

    def _initialise_client(self, url: str, token: str, timeout: float) -> None:
        self._url = url
        self._token = token
        self._timeout = timeout
        self.client = httpx.Client(
            base_url=self.url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "User-Agent": f"pipeline-ai/{package_version()}",
            },
            timeout=self._timeout,
        )

    @property
    def token(self):
        return self._token

    @property
    def url(self):
        return self._url

    def authenticate(self) -> None:
        """Authenticate with the pipeline.ai API."""
        if self.verbose:
            print("Authenticating")

        if self.token is None:
            raise MissingActiveToken(
                token="",
                message="Please pass a valid token or set it as an ENV var",
            )

        response = self.client.get("/v2/users/me")
        if response.status_code in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}:
            raise MissingActiveToken(token=self.token)
        else:
            self._get_raise_for_status(response)

        if self.verbose:
            print("Succesfully authenticated with the Pipeline API (%s)" % self.url)
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
    def _get_raise_for_status(response: httpx.Response) -> None:
        # A handler for errors that might be sent with messages from Top.
        if (
            isinstance(response, httpx.Response)
            and not response.status_code == httpx.codes.OK
        ):
            try:
                content = response.json()
            except json.JSONDecodeError:
                response.raise_for_status()

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

    def upload_data(self, obj: Any) -> DataGet:
        """Upload the object as a pickle-encoded file to the API."""
        files = dict(data=_as_upload_file(obj))
        uploaded_data = self._post("/v2/data", files=files)
        return DataGet.parse_obj(uploaded_data)

    def _initialise_direct_pipeline_file_upload(self, file_size: int) -> str:
        """Initialise a direct multi-part pipeline file upload"""
        direct_upload_schema = PipelineFileDirectUploadInitCreate(
            file_size=file_size, file_format=FileFormat.binary
        )
        response = self._post(
            "/v2/pipeline-files/initiate-multipart-upload",
            json_data=direct_upload_schema.dict(),
        )
        direct_upload_get = PipelineFileDirectUploadInitGet.parse_obj(response)
        return direct_upload_get.pipeline_file_id

    def _direct_upload_pipeline_file_chunk(
        self,
        data: Union[io.BytesIO, CallbackIOWrapper],
        pipeline_file_id: str,
        part_num: int,
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
            "/v2/pipeline-files/presigned-url",
            json_data=part_upload_schema.dict(),
        )
        part_upload_get = PipelineFileDirectUploadPartGet.parse_obj(response)
        # upload file chunk
        response = httpx.put(
            part_upload_get.upload_url,
            content=data,
            timeout=self._timeout,
        )
        response.raise_for_status()
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
            json_data=finalise_upload_schema.dict(),
        )
        return PipelineFileGet.parse_obj(response)

    def upload_pipeline_file(
        self, pipeline_file: PipelineFile
    ) -> PipelineFileVariableGet:
        """Upload PipelineFile given by pipeline_file.

        Since PipelineFiles can be very large, we implement this slightly
        differently to regular file uploads:
        - We need to split the file into chunks based on FILE_CHUNK_SIZE
        - We first initialise the multi-part upload with the server
        - We then upload the file in chunks (requesting a presigned upload URL for each
            chunk beforehand)
        - Lastly, we finalise the multi-part upload with the server
        """

        file_size = os.path.getsize(pipeline_file.path)

        pipeline_file_id = self._initialise_direct_pipeline_file_upload(
            file_size=file_size
        )

        parts = []
        file_hash = hashlib.sha256()
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
                file_hash.update(file_data)
                part_num = len(parts) + 1
                # If verbose then wrap our data object in a tqdm callback
                if self.verbose:
                    data = CallbackBytesIO(progress.update, file_data)
                else:
                    data = io.BytesIO(file_data)

                upload_metadata = self._direct_upload_pipeline_file_chunk(
                    data=data,
                    pipeline_file_id=pipeline_file_id,
                    part_num=part_num,
                )
                parts.append(upload_metadata)

        file_hash = file_hash.hexdigest()
        pipeline_file_get = self._finalise_direct_pipeline_file_upload(
            pipeline_file_id=pipeline_file_id, multipart_metadata=parts
        )
        return PipelineFileVariableGet(
            path=pipeline_file.path, file=pipeline_file_get.file, hash=file_hash
        )

    def _get(self, endpoint: str, params: Dict[str, Any] = None):
        self.raise_for_invalid_token()
        response = self.client.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()

    def _post(
        self,
        endpoint: str,
        json_data: dict = None,
        files: Optional[Union[list, dict]] = None,
    ) -> dict:
        self.raise_for_invalid_token()
        schema = json_data

        progresses = []
        data = None
        if files is not None:
            # Normalise dict/list types
            if isinstance(files, dict):
                files = files.items()
            files_payload = []
            for (form_name, (file_name, file_handle, file_type)) in files:
                # If verbose then wrap our file object in a tqdm callback
                if self.verbose:
                    progress = tqdm(
                        desc=f"{PIPELINE_STR} Uploading",
                        unit="B",
                        unit_scale=True,
                        total=file_handle.getbuffer().nbytes,
                        unit_divisor=1024,
                    )
                    file_handle = CallbackIOWrapper(progress.update, file_handle)
                    progresses.append(progress)
                files_payload.append((form_name, (file_name, file_handle, file_type)))
            files = files_payload

            # The `json` argument is ignored if `files` are given; we must
            # JSON-encode the data ourselves and pass it to the `data` argument.
            if json_data is not None:
                data = dict(json=json.dumps(json_data))
                json_data = None

        response = self.client.post(
            endpoint,
            json=json_data,
            data=data,
            files=files,
        )

        for progress in progresses:
            progress.close()

        if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
            raise InvalidSchema(schema=schema)
        else:
            self._get_raise_for_status(response)
        return response.json()

    def _patch(self, endpoint: str, json_data: dict) -> dict:
        self.raise_for_invalid_token()
        response = self.client.patch(endpoint, json=json_data)
        if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
            schema = json_data
            raise InvalidSchema(schema=schema)
        else:
            self._get_raise_for_status(response)
        return response.json()

    def _delete(self, endpoint: str) -> None:
        self.raise_for_invalid_token()
        response = self.client.delete(endpoint)
        if response.status_code != HTTPStatus.NO_CONTENT:
            self._get_raise_for_status(response)

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
            function_create_schema = FunctionCreate(
                local_id=function.local_id,
                name=function.name,
                function_source=function.source,
                hash=function.hash,
                inputs=inputs,
                output=output,
            )
        except AttributeError as e:
            raise InvalidSchema(schema="Function", message=str(e))

        response = self._post(
            "/v2/functions",
            files=dict(pickle=_as_upload_file(function.function)),
            json_data=function_create_schema.dict(),
        )
        return FunctionGet.parse_obj(response)

    def upload_model(self, model: Model) -> ModelGet:
        try:
            model_create_schema = ModelCreate(
                local_id=model.local_id,
                name=model.name,
                model_source=model.source,
                hash=model.hash,
            )
        except ValidationError as e:
            raise InvalidSchema(schema="Model", message=str(e))

        response = self._post(
            "/v2/models",
            files=dict(pickle=_as_upload_file(model.model)),
            json_data=model_create_schema.dict(),
        )
        return ModelGet.parse_obj(response)

    def upload_pipeline(
        self,
        new_pipeline_graph: Graph,
        public: bool = False,
        description: str = "",
        tags: Set[str] = None,
        environment: Optional[EnvironmentObjectOrID] = None,
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
                    environment (Optional[Union[PipelineCloudEnvironment, str]]):
                        Identifier of the execution environment the pipeline
                        should run within. If None (the default) a Mystic-
                        provided default environment will be chosen.

            Returns:
                    pipeline (PipelineGet): Object representing uploaded pipeline.
        """
        if new_pipeline_graph._has_run_startup:
            raise Exception("Cannot upload a pipeline that has already been run.")
        # Pipeline Cloud currently supports Python 3.9.x only
        if (sys.version_info.major, sys.version_info.minor) != (3, 9):
            print(
                "WARNING: pipeline-ai is still in development and the"
                " upload_pipeline function has only been tested in Python 3.9. "
                "We strongly recommend you use Python 3.9 as pipelines uploaded"
                " in other Python versions are known to be broken. We are working"
                "on adding support for 3.10 and 3.8!"
            )

        new_name = new_pipeline_graph.name
        if new_pipeline_graph.functions and self.verbose:
            print("Uploading functions")
        new_functions = [
            self.upload_function(_function)
            for _function in new_pipeline_graph.functions
        ]
        for i, uploaded_function_schema in enumerate(new_functions):
            new_pipeline_graph.functions[i].local_id = uploaded_function_schema.id
        if new_pipeline_graph.models and self.verbose:
            print("Uploading models")
        new_models = [self.upload_model(_model) for _model in new_pipeline_graph.models]

        new_variables: List[PipelineVariableCreate] = []
        variable_type_uploads = []
        for _var in new_pipeline_graph.variables:
            pipeline_file_schema = None
            if isinstance(_var, PipelineFile):
                if _var.remote_id is not None:
                    file_schema: FileGet = self._download_schema(
                        schema=FileGet,
                        endpoint=f"/v2/files/{_var.remote_id}",
                        params=dict(
                            return_data=False,
                        ),
                    )
                    unique_identifier = str(uuid.uuid4())
                    pipeline_file_schema = PipelineFileVariableGet(
                        path=unique_identifier, hash=unique_identifier, file=file_schema
                    )
                else:
                    pipeline_file_schema = self.upload_pipeline_file(_var)

            _var_schema = PipelineVariableCreate(
                local_id=_var.local_id,
                name=_var.name,
                is_input=_var.is_input,
                is_output=_var.is_output,
                pipeline_file_variable=pipeline_file_schema,
            )

            variable_type_uploads.append(
                _as_upload_file(
                    _var.type_class,
                    name=_var.local_id,
                )
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
                environment_id=resolve_environment_id(environment),
            )
        except ValidationError as e:
            raise InvalidSchema(schema="Graph", message=str(e))

        if self.verbose:
            print("Uploading pipeline graph")

        response = self._post(
            "/v2/pipelines",
            files=[("variable_types", v) for v in variable_type_uploads],
            json_data=json.loads(pipeline_create_schema.json()),
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
        if not isinstance(raw_data_or_schema, DataGet):
            uploaded_data = self.upload_data(raw_data_or_schema)
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
        run_json: dict = self._post(
            "/v2/runs",
            json_data=json.loads(run_create_schema.json()),
        )
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
        return load_object(d_get_schema.hex_file.data)

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
        return load_object(f_get_schema.data)

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

    def get_runs(
        self, limit: int = 20, skip: int = 0, created_at_order: str = "desc"
    ) -> List[RunGet]:
        # Get runs from the remote compute service
        result = self._get(
            "/v2/runs",
            params=dict(
                limit=limit, skip=skip, order_by=f"created_at:{created_at_order}"
            ),
        )
        return result

    def download_remotes(self, graph: Graph) -> None:
        # Only remote PipelineFiles are supported
        for variable in graph.variables:
            if not isinstance(variable, PipelineFile) or variable.remote_id is None:
                continue

            downloaded_schema: FileGet = self._download_schema(
                schema=FileGet,
                endpoint=f"/v2/files/{variable.remote_id}",
                params=dict(
                    return_data=True,
                ),
            )

            configuration.PIPELINE_CACHE_FILES.mkdir(exist_ok=True)
            raw_result = load_object(downloaded_schema.data)
            file_path = configuration.PIPELINE_CACHE_FILES / variable.remote_id
            with open(file_path, "wb") as file:
                dill.dump(raw_result, file=file)

            variable.path = file_path
