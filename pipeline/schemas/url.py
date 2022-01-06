import ipaddress

from .base import BaseModel


class URLBase(BaseModel):
    url_string: str
    static_public_ip: ipaddress.IPv4Address


class URLCreate(URLBase):
    pass


class URLGet(URLBase):
    id: str
