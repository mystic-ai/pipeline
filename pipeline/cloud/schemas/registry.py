from pydantic import BaseModel


class RegistryInformation(BaseModel):
    url: str
    special_auth: bool
