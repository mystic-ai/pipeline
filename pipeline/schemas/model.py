from typing import Optional

from pipeline.schemas.base import BaseModel


class ModelBase(BaseModel):
    id: Optional[str]
    name: str
