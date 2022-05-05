from pipeline.schemas.validators import valid_password

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


def test_invalid_password():
    for password in INVALID_PASSWORDS:
        assert valid_password(password) == False


def test_valid_password():
    for password in VALID_PASSWORDS:
        assert valid_password(password) == True
