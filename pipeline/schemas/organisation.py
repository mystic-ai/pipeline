from typing import Optional

from .base import AvatarHolder, Patchable


class OrganisationBase(AvatarHolder):
    name: str


class OrganisationGet(OrganisationBase):
    id: str
    member_count: int


class OrganisationCreate(OrganisationBase):
    pass


class OrganisationPatch(OrganisationBase, Patchable):
    name: Optional[str]
