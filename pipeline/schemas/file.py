from typing import List, Optional

from .base import BaseModel


class FileBase(BaseModel):
    name: str


class FileGet(FileBase):
    id: str
    path: str
    #: The data as hex-encoded bytes, if the data size is less than 20 kB
    data: Optional[str]
    #: The data size in kilobytes (kB)
    file_size: int


class FileCreate(FileBase):
    name: Optional[str]
    file_bytes: Optional[str]


class FileDirectUploadInitCreate(FileBase):
    file_size: int


class FileDirectUploadInitGet(BaseModel):
    upload_id: str
    file_id: str


class FileDirectUploadPartCreate(BaseModel):
    upload_id: str
    file_id: str
    part_num: int


class FileDirectUploadPartGet(BaseModel):
    upload_url: str
    # url_expiry_time: datetime


class FileDirectUploadFinaliseCreate(BaseModel):
    upload_id: str
    file_id: str
    multipart_metadata: List[dict]
