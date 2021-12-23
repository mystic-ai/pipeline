from typing import List

from .base import BaseModel
from .resource import ResourceCreate, ResourceGet


class WorkerBase(BaseModel):
    worker_ip: str
    worker_name: str


class WorkerCreate(WorkerBase):
    resources: List[ResourceCreate]


class WorkerGet(WorkerBase):
    id: str
    resources: List[ResourceGet]
