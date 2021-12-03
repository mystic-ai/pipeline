from pydantic import Field

from .base import AvatarHolder, BaseModel, Patchable


class OrganisationMemberBase(BaseModel):
    email: str


class OrganisationMemberCreate(OrganisationMemberBase):
    pass


class OrganisationMemberUserGet(AvatarHolder):
    id: str
    email: str
    username: str


class OrganisationMemberGet(BaseModel):
    organisation_id: str
    role: str = Field(alias="role_name")
    user: OrganisationMemberUserGet


class OrganisationMemberPatch(Patchable):
    role: str
