import enum
from typing import Optional, Union

from pydantic import StrictBytes, StrictStr

from .base import BaseModel


class FileFormat(str, enum.Enum):
    """Represents the different formats files can be uploaded in"""

    hex = "hex"
    binary = "binary"


class FileBase(BaseModel):
    name: str
    # hex is the default purely for backwards-compatability
    file_format: FileFormat = FileFormat.hex


class FileGet(FileBase):
    id: str
    path: str
    #: When str, the data are hex-encoded bytes, else plain bytes
    data: Optional[Union[StrictBytes, StrictStr]]
    #: The data size in bytes
    file_size: int


class FileCreate(FileBase):
    name: Optional[str]
    file_bytes: Optional[str]
