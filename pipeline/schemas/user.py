from typing import List, Optional

from pydantic import validator

from .base import AvatarHolder, BaseModel, Patchable
from .token import TokenGet
from .validators import valid_email, valid_password, valid_username


class UserBase(AvatarHolder):
    email: str
    username: str
    firstname: Optional[str]
    lastname: Optional[str]
    company: Optional[str]
    job_title: Optional[str]


class UserGet(UserBase):
    id: str
    oauth_provider: Optional[str]
    verified: Optional[bool]
    subscribed: Optional[bool]


class UserGetDetailed(UserGet):
    tokens: List[TokenGet] = []


class UserGetEnriched(UserGetDetailed):
    base_token: TokenGet


class UserPatch(Patchable, AvatarHolder):
    firstname: Optional[str]
    lastname: Optional[str]
    company: Optional[str]
    job_title: Optional[str]
    subscribed: Optional[bool]


class UserUsernamePatch(Patchable):
    username: str

    @validator("username")
    def validate_username(cls, value):
        if not valid_username(value):
            raise ValueError(
                (
                    "must contain between 3-24 characters, only alphanumerics, "
                    "hyphens and underscores."
                )
            )
        return value


class UserEmailPatch(Patchable):
    email: str

    @validator("email")
    def validate_email(cls, value):
        lowered_value = value.lower()
        if not valid_email(lowered_value):
            raise ValueError("doesn't match standard email format")
        return lowered_value


class UserPasswordPatch(Patchable):
    old_password: str
    password: str

    @validator("password")
    def validate_password(cls, value):
        if not valid_password(value):
            raise ValueError(
                (
                    "must contain at least 8 characters, "
                    "one uppercase letter and one number."
                )
            )
        return value


class UserPasswordResetPatch(Patchable):
    password: str

    @validator("password")
    def validate_password(cls, value):
        if not valid_password(value):
            raise ValueError(
                (
                    "must contain at least 8 characters, "
                    "one uppercase letter and one number."
                )
            )
        return value


class UserLogin(BaseModel):
    email: str
    password: str

    @validator("email")
    def validate_email(cls, value):
        lowered_value = value.lower()
        if not valid_email(lowered_value):
            raise ValueError("doesn't match standard email format")
        return lowered_value


class UserOAuthLogin(BaseModel):
    email: str
    oauth_id: str
    oauth_provider: str


class UserCreate(UserBase):
    password: str
    username: Optional[str]
    account_type: Optional[str]

    @validator("email")
    def validate_email(cls, value):
        lowered_value = value.lower()
        if not valid_email(lowered_value):
            raise ValueError("doesn't match standard email format")
        return lowered_value

    @validator("password")
    def validate_password(cls, value):
        if not valid_password(value):
            raise ValueError(
                (
                    "must contain at least 8 characters, "
                    "one uppercase letter and one number."
                )
            )
        return value

    @validator("username")
    def validate_username(cls, value):
        if not valid_username(value):
            raise ValueError(
                (
                    "must contain between 3-24 characters, "
                    "only alphanumerics, hyphens and underscores."
                )
            )
        return value
