from .base import BaseModel

"""
Dependencies of a runnable/run/model etc..
e.g. pytorch v2.0.1, numpy v1.4
"""


class DependencyBase(BaseModel):
    name: str


class DependencyCreate(DependencyBase):
    pass


class DependencyGet(DependencyBase):
    id: str
