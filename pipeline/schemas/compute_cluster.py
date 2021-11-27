from .base import BaseModel
from .url import URLGet


class ComputeClusterBase(BaseModel):
    compute_cluster_name: str
    compute_cluster_cloud: str


class ComputeClusterGet(ComputeClusterBase):
    id: str
    url: URLGet


class ComputeClusterCreate(ComputeClusterBase):
    compute_cluster_controller_url: str
