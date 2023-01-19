from typing import Optional

from .base import AvatarHolder, Patchable


class ProjectBase(AvatarHolder):
    name: str


class ProjectCreate(ProjectBase):
    pass


class ProjectGet(ProjectBase):
    id: str


class ProjectGetDetailed(ProjectGet):
    """Include counts of 'held' resources."""

    n_functions: int
    n_pipelines: int
    n_models: int
    n_function_runs: int
    n_pipeline_runs: int
    n_data: int

    class Config:
        orm_mode = False


class ProjectPatch(ProjectBase, Patchable):
    name: Optional[str]
