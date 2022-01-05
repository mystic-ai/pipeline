from uuid import uuid4

from pydantic import BaseModel, Field


class RedisCommandSchema(BaseModel):
    command_thread_id: str = Field(default_factory=lambda: str(uuid4()))
    command: str
    data: dict
    to: str
    sender: str
