from typing import Optional

from pipeline.schemas.base import BaseModel
from pipeline.schemas.pipeline import PipelineGet
from pipeline.schemas.project import ProjectGet


class DeploymentCreate(BaseModel):
    """Schema for creating a pipeline deployment"""

    pipeline_id: str
    # If not provided, deployment will be created with users' default project
    project_id: Optional[str] = None


class DeploymentGet(BaseModel):
    id: str
    pipeline: PipelineGet
    project: ProjectGet
    active: bool


class DeploymentPatch(BaseModel):
    active: Optional[bool]
