import io
import os
from pathlib import Path

import httpx
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper

from pipeline.cloud import http
from pipeline.cloud.schemas import files as s
from pipeline.util import CallbackBytesIO
from pipeline.util.logging import PIPELINE_FILE_STR

FILE_CHUNK_SIZE = 200 * 1024 * 1024  # 200 MiB


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
    # upload file chunk
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