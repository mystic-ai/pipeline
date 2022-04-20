from typing import List, Optional

from pydantic import Field, root_validator

from pipeline.schemas.base import BaseModel
from pipeline.schemas.pipeline import PipelineGet


class DeploymentGet(BaseModel):
    project_id: str


class DeploymentGet(DeploymentGet):
    pipeline: PipelineGet
