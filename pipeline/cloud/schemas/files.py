from datetime import datetime

from pipeline.cloud.schemas import BaseModel


class RemoteFileData(BaseModel):
    id: str
    path: str
    url: str | None


class FileGet(BaseModel):
    id: str
    path: str

    created_at: datetime
    updated_at: datetime


class MultipartFileUploadInitCreate(BaseModel):
    name: str | None = None


class MultipartFileUploadInitGet(BaseModel):
    file_id: str
    upload_id: str


class MultipartFileUploadPartCreate(BaseModel):
    file_id: str
    upload_id: str
    # The part number for this multi-part file upload
    part_num: int


class MultipartFileUploadPartGet(BaseModel):
    # The URL to use when uploading the file chunk
    upload_url: str


class MultipartFileUploadMetadata(BaseModel):
    """Schema for multi-part uploads direct to storage server.

    (We don't have any control over this schema)
    """

    ETag: str
    PartNumber: int


class MultipartFileUploadFinaliseCreate(BaseModel):
    file_id: str
    upload_id: str
    # The metadata obtained from each part of the file upload
    multipart_metadata: list[MultipartFileUploadMetadata]


class UploadFileUsingPresignedUrl(BaseModel):
    local_file_path: str
    upload_url: str
    upload_fields: dict[str, str]


class UploadFilesToRemoteStorageCreate(BaseModel):
    files: list[UploadFileUsingPresignedUrl]
