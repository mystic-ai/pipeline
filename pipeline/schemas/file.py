from typing import Optional

from datetime import datetime

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


class FileDirectUploadCreate(FileBase):
    file_hash: Optional[str]
    file_size: int


class FileDirectUploadGet(BaseModel):
    upload_id: str
    # upload_fields: dict
    # url_expiry_time: datetime
    # file_id: str
