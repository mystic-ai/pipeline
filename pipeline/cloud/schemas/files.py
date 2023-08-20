from datetime import datetime

from pipeline.cloud.schemas import BaseModel


class FileGet(BaseModel):
    id: str
    path: str

    created_at: datetime
    updated_at: datetime
