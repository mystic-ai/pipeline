from typing import List

from .base import BaseModel
from .file import FileGet


class PipelineFileDirectUploadInitCreate(BaseModel):
    file_size: int


class PipelineFileDirectUploadInitGet(BaseModel):
    pipeline_file_id: str


class PipelineFileDirectUploadPartCreate(BaseModel):
    pipeline_file_id: str
    # The part number for this multi-part file upload
    part_num: int


class PipelineFileDirectUploadPartGet(BaseModel):
    # The URL to use when uploading the fie
    upload_url: str


class MultipartUploadMetadata(BaseModel):
    """Schema for multi-part uploads direct to storage server.

    (We don't have any control over this schema)
    """

    ETag: str
    PartNumber: int


class PipelineFileDirectUploadFinaliseCreate(BaseModel):
    pipeline_file_id: str
    # The metadata obtained from each part of the file upload
    multipart_metadata: List[MultipartUploadMetadata]


class PipelineFileGet(BaseModel):
    id: str
    name: str
    hex_file: FileGet
