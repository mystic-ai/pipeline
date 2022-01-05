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
# Simple password regex, requires a minimum of 8 characters, one uppercase letter,
# one lowercase letter and a number.
PASSWORD_REGEX = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[a-zA-Z\d]{8,}$")
# Simple username regex, number of characters between 3-24, allowing only alphanumerics,
# dashes and underscores.
USERNAME_REGEX = re.compile(r"^[a-zA-Z0-9-_]{3,24}$")


def valid_email(email_string: str) -> bool:
    return EMAIL_REGEX.match(email_string) is not None


def valid_password(password_string: str) -> bool:
    return PASSWORD_REGEX.match(password_string) is not None


def valid_username(username_string: str) -> bool:
    return USERNAME_REGEX.match(username_string) is not None
