from typing import Optional

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
