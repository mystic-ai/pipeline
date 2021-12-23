from typing import Optional

from .base import BaseModel


class RequestGet(BaseModel):
    id: str
    token: Optional[str]
    status: str
    request_json: str
    result_json: str
    result_file_url: Optional[str]
    resource_url: str
    request_method: str
    time_requested: float
    request_duration: int
    ip_address: str
