from typing import Optional

from .base import BaseModel


class ResourceBase(BaseModel):
    foreign_id: Optional[str]
    resource_label: str
    resource_type: str


class ResourceCreate(ResourceBase):
    pass


class ResourceGet(ResourceBase):
    id: str
