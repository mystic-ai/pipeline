from pydantic import BaseModel


class PipelineClusterConfig(BaseModel):
    id: str
    node_pool: str


class PipelineClusterLean(BaseModel):
    id: str
    name: str
    node_pool: str | None = None
    provider: str | None = None
