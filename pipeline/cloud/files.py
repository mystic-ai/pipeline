import io
import os
import uuid
from pathlib import Path
from zipfile import ZipFile

import httpx
from httpx import HTTPStatusError
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper

from pipeline.cloud import http
from pipeline.cloud.schemas import files as s
from pipeline.cloud.schemas.runs import RunInput, RunIOType
from pipeline.objects import Directory, File
from pipeline.util import CallbackBytesIO
from pipeline.util.logging import PIPELINE_FILE_STR

FILE_CHUNK_SIZE = 200 * 1024 * 1024  # 200 MiB


def upload_file(file_path: Path) -> s.RemoteFileData:
    """Upload file directly to storage using single-part upload."""
    with file_path.open("rb") as f:
        files = {"file": f}
        files = {"pfile": (f.name, f)}
        response = http.post_file("/v4/files", files=files)
        return s.RemoteFileData.parse_obj(response.json())


def upload_multipart_file(file_path: Path, progress: bool = False) -> s.FileGet:
    """Upload file directly to storage using multi-part upload.

    Since files can be very large (e.g. model weights), we use the following process:
    - We need to split the file into chunks based on FILE_CHUNK_SIZE
    - We first initialise the multi-part upload with the server
    - We then upload the file in chunks (requesting a presigned upload URL for each
        chunk beforehand)
    - Lastly, we finalise the multi-part upload with the server
    """

    file_size = os.path.getsize(file_path)
    init_result = _init_multipart_upload(file_path.name)
    file_id = init_result.file_id
    upload_id = init_result.upload_id

    progress_bar = None
    if progress:
        progress_bar = tqdm(
            desc=f"{PIPELINE_FILE_STR} Uploading {file_path}",
            unit="B",
            unit_scale=True,
            total=file_size,
            unit_divisor=1024,
        )
    parts = []
    with file_path.open("rb") as f:
        while True:
            file_data = f.read(FILE_CHUNK_SIZE)
            if not file_data:
                if progress_bar:
                    progress_bar.close()
                break
            part_num = len(parts) + 1
            # If displaying progress bar then wrap our data object in a tqdm callback
            if progress_bar:
                data = CallbackBytesIO(progress_bar.update, file_data)
            else:
                data = io.BytesIO(file_data)

            upload_metadata = _upload_multipart_file_chunk(
                data=data,
                file_id=file_id,
                upload_id=upload_id,
                part_num=part_num,
            )
            parts.append(upload_metadata)

    file_get = _finalise_multipart_upload(
        file_id=file_id, upload_id=upload_id, multipart_metadata=parts
    )
    return file_get


def is_file_like(obj):
    return isinstance(obj, File) or isinstance(obj, io.IOBase)


def _init_multipart_upload(filename: str) -> s.MultipartFileUploadInitGet:
    res = http.post(
        "/v3/pipeline_files/initiate-multipart-upload",
        s.MultipartFileUploadInitCreate(name=filename).dict(),
    )
    return s.MultipartFileUploadInitGet.parse_obj(res.json())


def _upload_multipart_file_chunk(
    data: io.BytesIO | CallbackIOWrapper,
    file_id: str,
    upload_id: str,
    part_num: int,
) -> s.MultipartFileUploadMetadata:
    """Upload a single chunk of a multi-part pipeline file upload.

    Returns the metadata associated with this upload (this is needed to pass into
    the finalisation step).
    """
    # get presigned URL
    part_upload_schema = s.MultipartFileUploadPartCreate(
        file_id=file_id,
        upload_id=upload_id,
        part_num=part_num,
    )

    res = http.post(
        "/v3/pipeline_files/presigned-url",
        part_upload_schema.dict(),
    )

    part_upload_get = s.MultipartFileUploadPartGet.parse_obj(res.json())

    response = httpx.put(
        part_upload_get.upload_url,
        content=data,
        timeout=60,
    )
    response.raise_for_status()
    etag = response.headers["ETag"]
    return s.MultipartFileUploadMetadata(ETag=etag, PartNumber=part_num)


def _finalise_multipart_upload(
    file_id: str,
    upload_id: str,
    multipart_metadata: list[s.MultipartFileUploadMetadata],
) -> s.FileGet:
    """Finalise the direct multi-part file upload"""
    finalise_upload_schema = s.MultipartFileUploadFinaliseCreate(
        file_id=file_id,
        upload_id=upload_id,
        multipart_metadata=multipart_metadata,
    )
    response = http.post(
        "/v3/pipeline_files/finalise-multipart-upload",
        json_data=finalise_upload_schema.dict(),
    )
    return s.FileGet.parse_obj(response.json())


def get_path_from_id(file_id: str) -> str:
    file_get_response = http.get(f"/v3/pipeline_files/{file_id}")
    if file_get_response is None:
        raise ValueError(f"Received no response for file_id {file_id}")

    file_get_schema = s.FileGet.parse_obj(file_get_response.json())

    return file_get_schema.path


def create_remote_directory(local_path: Path) -> s.FileGet:
    local_path_str = str(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Directory not found (path={local_path_str}) ")
    if not local_path.is_dir() and not local_path_str.endswith(".zip"):
        raise ValueError(f"Path is not a directory or zip file (path={local_path_str})")

    zip_path = local_path

    if not local_path_str.endswith(".zip"):
        tmp_path = Path("/tmp") / (str(uuid.uuid4()) + ".zip")
        with ZipFile(str(tmp_path), "w") as zip_file:
            for root, dirs, files in os.walk(local_path_str):
                for file in files:
                    zip_file.write(
                        os.path.join(root, file),
                        arcname=file,
                    )
        zip_path = tmp_path

    try:
        file_get = upload_multipart_file(zip_path, progress=True)
    except HTTPStatusError as e:
        if e.response.status_code == 403:
            raise Exception(
                f"Permission denied uploading directory (path={local_path_str})"
            )
        raise Exception(f"Error uploading directory (path={local_path_str}): {e}")

    return file_get


def resolve_pipeline_file_object(obj: File | Directory) -> None:
    # Handle from ID, URL, or local path
    # Either URL or path has to be popluated, and on the remote

    if obj.url is not None:
        return
    elif obj.remote_id is not None:
        obj.path = Path(get_path_from_id(obj.remote_id))
        return
    if obj.path:
        if isinstance(obj, Directory):
            remote_dir = create_remote_directory(obj.path)
            obj.path = Path(remote_dir.path)
            return
        elif isinstance(obj, File):
            remote_file = upload_multipart_file(obj.path)
            obj.path = Path(remote_file.path)
            return


def resolve_run_input_file_object(obj: File | Directory) -> RunInput:
    if obj.url is not None:
        return RunInput(
            type=RunIOType.file,
            value=None,
            file_name=obj.url.geturl().split("/")[-1],
            file_path=obj.url.geturl(),
        )
    elif obj.remote_id is not None:
        path = get_path_from_id(obj.remote_id)
        return RunInput(
            type=RunIOType.file,
            value=None,
            file_name=path.split("/")[-1],
            file_path=path,
        )
    elif obj.path is not None:
        if isinstance(obj, Directory):
            remote_dir = create_remote_directory(obj.path)
            return RunInput(
                type=RunIOType.file,
                value=None,
                file_name=remote_dir.path.split("/")[-1],
                file_path=remote_dir.path,
            )
        elif isinstance(obj, File):
            remote_file = upload_file(obj.path)
            return RunInput(
                type=RunIOType.file,
                value=None,
                file_name=remote_file.path.split("/")[-1],
                file_path=remote_file.path,
                file_url=remote_file.url,
            )

    raise Exception(f"Invalid file object: {obj}, must have remote_id, path, or URL")
