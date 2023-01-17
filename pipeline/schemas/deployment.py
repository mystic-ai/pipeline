from typing import Optional

from pipeline.schemas.base import BaseModel
from pipeline.schemas.pipeline import PipelineGet


class DeploymentCreate(BaseModel):
    """Schema for creating a pipeline deployment"""

    pipeline_id: str


class DeploymentGet(BaseModel):
    id: str
    pipeline: PipelineGet
    active: bool


class DeploymentPatch(BaseModel):
    active: Optional[bool]
