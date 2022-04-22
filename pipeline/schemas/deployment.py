from typing import List, Optional

from pydantic import Field, root_validator

from pipeline.schemas.base import BaseModel
from pipeline.schemas.pipeline import PipelineGet


class DeploymentCreate(BaseModel):
    project_id: str


class DeploymentGet(DeploymentCreate):
    pipeline: PipelineGet
