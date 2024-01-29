from pydantic import BaseModel


class PipelineClusterConfig(BaseModel):
    id: str
    node_pool: str
