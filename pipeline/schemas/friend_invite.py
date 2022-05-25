from enum import Enum


class FriendInviteStatus(Enum):
    CREATED = "created"
    # Friend has accepted the invite
    ACCEPTED = "accepted"
    # The friend is still under trial period
    TRIAL = "trial"
    # Cycle complete, the inviter has been credited
    COMPLETE = "complete"
