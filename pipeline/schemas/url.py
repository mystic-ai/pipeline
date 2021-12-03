import ipaddress

from pydantic import AnyHttpUrl

from .base import BaseModel


class URLBase(BaseModel):
    url_string: AnyHttpUrl
    static_public_ip: ipaddress.IPv4Address


class URLCreate(URLBase):
    pass


class URLGet(URLBase):
    id: str
