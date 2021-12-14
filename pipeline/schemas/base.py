import re
from datetime import datetime
from typing import Optional

import humps
from pydantic import BaseModel as PydanticModel
from pydantic import Extra, validator
from pydantic.generics import GenericModel as PydanticGenericModel

#: When set to True, model fields can be get/set by camelCase'd field names
#: This is useful for JS interop., where camelCase is the convention
#: (rather than Python's snake_case convention)
CAMEL_CASE_ALIASES = False


def _generate_alias(s):
    return humps.camelize(s) if CAMEL_CASE_ALIASES else s


class BaseConfig:
    alias_generator = _generate_alias
    allow_population_by_field_name = True
    # Encode datetime objects to JSON as integer timestamps
    json_encoders = {datetime: datetime.timestamp}
    orm_mode = True


class BaseModel(PydanticModel):
    """Base model for schemas."""

    class Config(BaseConfig):
        pass


class GenericModel(PydanticGenericModel):
    """Base model for schemas with generic typing."""

    class Config(BaseConfig):
        pass


class Patchable(BaseModel):
    """Tag type for marking schemas for patching.

    A Patchable schema is one used for updating other schemas.
    """

    class Config:
        # Forbid fields not specified in the model; this prevents silently
        # failing updates to patch non-matching fields
        extra = Extra.forbid


class AvatarHolder(BaseModel):
    """Schema mixin for models which may hold avatars."""

    avatar_colour: Optional[str]
    avatar_image_url: Optional[str]

    @validator("avatar_colour")
    def validate_avatar_colour(cls, value: Optional[str]):
        if value is not None and not re.match(r"#[0-9a-f]{6}", value.lower()):
            raise ValueError("not a valid #rrggbb colour")
        return value
