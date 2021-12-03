from .base import BaseModel


class TagBase(BaseModel):
    name: str


class TagCreate(TagBase):
    pass


class TagGet(TagBase):
    id: str
    frequency: int = 0
