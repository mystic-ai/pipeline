from __future__ import annotations

import io
import json
import urllib.parse
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Type, Union

import httpx

from pipeline.exceptions.InvalidSchema import InvalidSchema
from pipeline.schemas.base import BaseModel
from pipeline.schemas.compute_requirements import ComputeRequirements
from pipeline.schemas.data import DataGet
from pipeline.schemas.file import FileCreate, FileGet
from pipeline.schemas.function import FunctionGet
from pipeline.schemas.model import ModelGet
from pipeline.schemas.pipeline import PipelineFileVariableGet, PipelineGet
from pipeline.schemas.pipeline_file import MultipartUploadMetadata, PipelineFileGet
from pipeline.schemas.run import RunCreate, RunGet
from pipeline.util import generate_id, python_object_to_hex

if TYPE_CHECKING:
    from pipeline.objects import Function, Graph, Model

from pipeline.api import PipelineCloud as _SyncPipelineCloud

FILE_CHUNK_SIZE = 200 * 1024 * 1024  # 200 MiB


class PipelineCloud(_SyncPipelineCloud):
    def _raise_not_implemeneted(self):
        raise NotImplementedError("This function is not implemented")

    def upload_python_object_to_file(self, obj, remote_path) -> FileGet:
        self._raise_not_implemeneted()

    def _initialise_direct_pipeline_file_upload(self, file_size: int) -> str:
        self._raise_not_implemeneted()

    def _direct_upload_pipeline_file_chunk(
        self, data: bytes, pipeline_file_id: str, part_num: int
    ) -> MultipartUploadMetadata:
        self._raise_not_implemeneted()

    def _finalise_direct_pipeline_file_upload(
        self, pipeline_file_id: str, multipart_metadata: List[MultipartUploadMetadata]
    ) -> PipelineFileGet:
        self._raise_not_implemeneted()

    def upload_pipeline_file(self, pipeline_file) -> PipelineFileVariableGet:
        self._raise_not_implemeneted()

    def _get(self, endpoint: str, params: Dict[str, Any] = None):
        self._raise_not_implemeneted()

    def upload_function(self, function: Function) -> FunctionGet:
        self._raise_not_implemeneted()

    def upload_model(self, model: Model) -> ModelGet:
        self._raise_not_implemeneted()

    def upload_pipeline(
        self,
        new_pipeline_graph: Graph,
        public: bool = False,
        description: str = "",
        tags: Set[str] = None,
    ) -> PipelineGet:
        self._raise_not_implemeneted()

    def _download_schema(
        self, schema: Type[BaseModel], endpoint: str, params: Optional[Dict[str, Any]]
    ) -> Type[BaseModel]:
        self._raise_not_implemeneted()

    def download_function(self, id: str) -> Function:
        self._raise_not_implemeneted()

    def download_model(self, id: str) -> Model:
        self._raise_not_implemeneted()

    def download_data(self, id: str) -> Any:
        self._raise_not_implemeneted()

    def download_result(self, result_id_or_schema: Union[str, RunGet]) -> Any:
        self._raise_not_implemeneted()

    def download_pipeline(self, id: str) -> Graph:
        self._raise_not_implemeneted()

    async def _post_file(self, endpoint, file, remote_path) -> FileGet:
        if not hasattr(file, "name"):
            file.name = generate_id(20)

        headers = {
            "Authorization": "Bearer %s" % self.token,
        }
        url = urllib.parse.urljoin(self.url, endpoint)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=headers,
                files={"file": (file.name, file, "application/octet-stream")},
                timeout=self.timeout,
            )
        if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
            schema = FileCreate.__name__
            raise InvalidSchema(schema=schema)
        else:
            response.raise_for_status()
        return FileGet.parse_obj(response.json())

    async def upload_file(self, file_or_path, remote_path) -> FileGet:

        if isinstance(file_or_path, str):
            # TODO: Change this to wrap the file object reader to convert to hex
            # everytime anything is read instead of reading it all at once.

            with open(file_or_path, "rb") as file:
                buffer = file.read()
            hex_buffer = buffer.hex()
            return await self._post_file(
                "/v2/files/", io.BytesIO(hex_buffer.encode()), remote_path
            )
        else:
            return await self._post_file("/v2/files/", file_or_path, remote_path)

    async def upload_data(self, file_or_path, remote_path) -> DataGet:
        uploaded_file = await self.upload_file(file_or_path, remote_path)
        uploaded_data = await self._post("/v2/data", uploaded_file.dict())
        return DataGet.parse_obj(uploaded_data)

    async def _post(self, endpoint, json_data):
        headers = {
            "Authorization": "Bearer %s" % self.token,
            "Content-type": "application/json",
        }

        url = urllib.parse.urljoin(self.url, endpoint)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url, headers=headers, json=json_data, timeout=self.timeout
            )

        if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
            schema = json_data
            raise InvalidSchema(schema=schema)
        else:
            self._get_raise_for_status(response)

        return response.json()

    async def run_pipeline(
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
            uploaded_data = await self.upload_data(temp_file, "/")
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
        run_json: dict = await self._post(
            "/v2/runs", json.loads(run_create_schema.json())
        )
        return RunGet.parse_obj(run_json)
