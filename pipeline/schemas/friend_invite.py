from enum import Enum

from .base import BaseModel


class FriendInviteStatus(Enum):
    CREATED = "created"
    #: Friend has accepted the invite
    ACCEPTED = "accepted"
    #: The friend is still under trial period
    TRIAL = "trial"
    #: Cycle complete, the inviter has been credited
    COMPLETE = "complete"


class FriendInviteBase(BaseModel):
    #: The ID of the User who sent the invite
    inviter_id: str
    #: The email the invite is to be sent to
    invitee_email: str


class FriendInviteCreate(FriendInviteBase):
    """Create an invitation for a friend to join"""

    pass


class FriendInviteGet(FriendInviteBase):
    """View of an invitation for a friend to join"""

    #: The ID of this invite
    id: str
    #: The status of the invite
    status: FriendInviteStatus


class FriendInvitePatch(BaseModel):
    """Patch the status of a friend invitation"""

    status: FriendInviteStatus
