import os
from datetime import datetime

import humps
from pydantic import BaseModel as _BaseModel
from pydantic import Extra

CAMEL_CASE_ALIASES = bool(os.environ.get("CAMEL_CASE_ALIASES", False))


def _generate_alias(s):
    return humps.camelize(s) if CAMEL_CASE_ALIASES else s


class BaseConfig:
    alias_generator = _generate_alias
    allow_population_by_field_name = True
    # Encode datetime objects to JSON as integer timestamps
    json_encoders = {datetime: datetime.timestamp}
    orm_mode = True


class BaseModel(_BaseModel):
    class Config(BaseConfig):
        orm_mode = True
        use_enum_values = True


class Patchable(BaseModel):
    class Config:
        extra = Extra.forbid
