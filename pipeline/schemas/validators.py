import re

# TO-DO: refactor validators as below
# https://pydantic-docs.helpmanual.io/usage/validators/

# Email regex mostly following RFC2822 specification. Covers ~99% of emails in use today
# Allows groups of alphanumerics and some special characters separated by dots,
# followed by a @,
# followed by groups of alphanumerics and non-staring/non-ending dashes,
# separated by dots.
EMAIL_REGEX = re.compile(
    r"[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*"
    r"@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)"
    r"+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"
)

# Simple password regex, requires a minimum of 8 characters with at least one
# uppercase letter, one lowercase letter, and one number.
PASSWORD_REGEX = re.compile(r"^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$")

# Simple username regex, number of characters between 3-24, allowing only alphanumerics,
# dashes and underscores.
USERNAME_REGEX = re.compile(r"^[a-zA-Z0-9-_]{3,24}$")

# Pipeline names may contain lowercase letters, digits and separators.
# Separators are periods, underscores, dashes, and forward slashes.
# They cannot start or end with a separator.
PIPELINE_NAME_REGEX = re.compile(r"^[a-z0-9][a-z0-9-._/]*[a-z0-9]$")

# We loosely follow the Docker tag conventions for valid tag names, specifically:
#
# - A tag name comprises a 'name' component and a 'tag' component in that order
#   separated by a colon, e.g. `name:tag`.
# - Name components may contain lowercase letters, digits and separators.
#   Separators are periods, underscores, dashes, and forward slashes.
#   A name component may not start or end with a separator.
# - A tag name must be valid ASCII and may contain lowercase and uppercase letters,
#   digits, underscores, periods and dashes. A tag name may not start with a period
#   or a dash and may contain a maximum of 128 characters.
PIPELINE_TAG_NAME_REGEX = re.compile(
    r"^[a-z0-9][a-z0-9-._/]*[a-z0-9]:[0-9A-Za-z_][0-9A-Za-z-_.]{0,127}$"
)


def valid_email(email_string: str) -> bool:
    return EMAIL_REGEX.match(email_string) is not None


def valid_password(password_string: str) -> bool:
    return PASSWORD_REGEX.match(password_string) is not None


def valid_username(username_string: str) -> bool:
    return USERNAME_REGEX.match(username_string) is not None


def valid_pipeline_name(name: str) -> bool:
    return PIPELINE_NAME_REGEX.match(name) is not None


def valid_pipeline_tag_name(name: str) -> bool:
    return PIPELINE_TAG_NAME_REGEX.match(name) is not None
