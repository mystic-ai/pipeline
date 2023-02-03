from pipeline.schemas.validators import valid_password, valid_username

INVALID_PASSWORDS = [
    "askjhs",  # < 8 chars
    "akljhakah",  # only letter
    "Qlkjhsslhs",  # no number
    "1987292hjs",  # no capital
]

VALID_PASSWORDS = [
    "akljhakah20982sJYUTG",
    "bU]2V2~bv!jp.%_R",
    "N?t,ATe/2!n@*))<DTr~+c[F=r",
    "''f^tTw3!dT={6_P)wSsFu_hcR:s6;De:6`_#6A!eKw//'exp9RAv.QdsVZ]fWWfxT!$tJST:7K",
]

INVALID_USERNAMES = [
    "as",  # < 3 chars
    "assj" * 7,  # > 24 chars
    "akljhak~!#ah",  # contains non-alphanumeric character which is neither "-" or "_"
]

VALID_USERNAMES = [
    "asakiuy",  # > 3 chars
    "ass6" * 6,  # <= 24 chars
    "akljhak-ah82_",  # allowed special chars
]


def test_invalid_password():
    for password in INVALID_PASSWORDS:
        assert not valid_password(password)


def test_valid_password():
    for password in VALID_PASSWORDS:
        assert valid_password(password)


def test_invalid_usernames():
    for username in INVALID_USERNAMES:
        assert not valid_username(username)


def test_valid_usernames():
    for username in VALID_USERNAMES:
        assert valid_username(username)
