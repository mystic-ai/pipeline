from typing import Optional

from pipeline.schemas.base import BaseModel
from pipeline.schemas.pipeline import PipelineGet
from pipeline.schemas.project import ProjectGet


class DeploymentCreate(BaseModel):
    project_id: str
    pipeline_id: str


class DefaultProjectDeploymentCreate(BaseModel):
    """Schema for creating a deployment on a users' default project"""
    pipeline_id: str


class DeploymentGet(BaseModel):
    id: str
    pipeline: PipelineGet
    project: ProjectGet
    active: bool


class DeploymentPatch(BaseModel):
    active: Optional[bool]
