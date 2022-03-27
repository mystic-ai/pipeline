import enum
from datetime import datetime
from typing import Optional

from .base import BaseModel
from .file import FileGet
from .token import TokenGet
from .project import ProjectGet


class FileType(enum.Enum):
    text = "text"
    image = "image"


class DataCreate(BaseModel):
    project_id: Optional[str]
    file: FileGet


class DataGet(BaseModel):
    id: str
    hex_file: FileGet
    created_at: datetime
    modified_at: Optional[datetime]
    name: Optional[str]
    size: Optional[int]
    file_type: Optional[FileType]
    token_created_by: Optional[TokenGet]
    token_modified_by: Optional[TokenGet]
    url: Optional[str]
    preview: Optional[str]
    project: Optional[ProjectGet]
