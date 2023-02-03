from .base import BaseModel


class OrganisationMemberInviteBase(BaseModel):
    #: The ID of the Organisation the invite relates to
    organisation_id: str
    #: The email the invite is to be sent to
    email: str


class OrganisationMemberInviteCreate(OrganisationMemberInviteBase):
    """Create an invitation to join an Organisation."""

    pass


class OrganisationMemberInviteGet(OrganisationMemberInviteBase):
    """View of an invitation to join an Organisation."""

    #: The ID of this invite
    id: str
    #: If True this invite has been accepted (emailed link has been visited)
    accepted: bool
    #: If True the invitation has been linked with an existing User
    has_user: bool
