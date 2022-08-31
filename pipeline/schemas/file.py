from typing import List, Optional

from .base import BaseModel


class FileBase(BaseModel):
    name: str


class FileGet(FileBase):
    id: str
    path: str
    #: The data as hex-encoded bytes, if the data size is less than 20 kB
    data: Optional[str]
    #: The data size in bytes
    file_size: int


class FileCreate(FileBase):
    name: Optional[str]
    file_bytes: Optional[str]


class MultiPartDirectFileUpload(BaseModel):
    """Base class for multipart direct file uploads"""

    upload_id: str
    file_id: str


class FileDirectUploadInitCreate(FileBase):
    file_size: int


class FileDirectUploadInitGet(MultiPartDirectFileUpload):
    pass


class FileDirectUploadPartCreate(MultiPartDirectFileUpload):
    # The part number for this multi-part file upload
    part_num: int


class FileDirectUploadPartGet(BaseModel):
    # The URL to use when uploading the fie
    upload_url: str


class FileDirectUploadFinaliseCreate(MultiPartDirectFileUpload):
    # The metadata obtained from each part of the file upload
    multipart_metadata: List[dict]
