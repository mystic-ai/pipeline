import pytest

from pipeline.schemas.validators import (
    valid_password,
    valid_pipeline_name,
    valid_pipeline_tag_name,
    valid_username,
)

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


@pytest.mark.parametrize(
    "pipeline_name",
    [
        "pipeline",
        "pipeline123",
        "pipeline.1233",
        "my-pipeline",
        "my_pipeline",
        "mystic/pipeline",
    ],
)
def test_valid_pipeline_name(pipeline_name):
    assert valid_pipeline_name(pipeline_name)


@pytest.mark.parametrize(
    "pipeline_name",
    [
        # empty
        "",
        # No blank space allowed
        # name starts or ends with a separator
        "-pipeline 123",
        "/pipeline1233",
        "my-pipeline_",
        # invalid characters
        "mystic(pipeline",
        "mystic#pipeline",
    ],
)
def test_invalid_pipeline_name(pipeline_name):
    assert not valid_pipeline_name(pipeline_name)


@pytest.mark.parametrize(
    "tag_name",
    [
        # Valid character set
        "pipeline:tag",
        "pipeline123:tag123",
        "pipeline.123:tag.123",
        "my-pipeline:my-tag",
        "my_pipeline:my_tag",
        "mystic/pipeline:tag",
        "mystic/pipeline:_tag",
    ],
)
def test_valid_pipeline_tag_name(tag_name):
    assert valid_pipeline_tag_name(tag_name)


@pytest.mark.parametrize(
    "tag_name",
    [
        # empty
        "",
        ":",
        # No tag
        "pipeline",
        "pipeline:",
        # No name
        ":tag",
        # Name starts or ends with a separator
        "_pipeline:tag",
        "-pipeline:tag",
        ".pipeline:tag",
        "/pipeline:tag",
        "pipeline_:tag",
        "pipeline-:tag",
        "pipeline.:tag",
        "pipeline/:tag",
        # Tag starts with a non-underscore separator
        "pipeline:-tag",
        "pipeline:.tag",
        # Invalid tag characters
        "pipeline:my/tag",
        # Tag too long
        "pipeline:" + ("a" * 129),
    ],
)
def test_invalid_pipeline_tag_name(tag_name):
    assert not valid_pipeline_tag_name(tag_name)
