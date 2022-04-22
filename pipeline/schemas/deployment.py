from typing import List, Optional

from pydantic import Field, root_validator

from pipeline.schemas.base import BaseModel
from pipeline.schemas.pipeline import PipelineGet
from pipeline.schemas.project import ProjectGet


class DeploymentCreate(BaseModel):
    project_id: str
    pipeline_id: str


class DeploymentGet(BaseModel):
    pipeline: PipelineGet
    project: ProjectGet
