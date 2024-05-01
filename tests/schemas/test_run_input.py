from pipeline.cloud.schemas.runs import RunInput, RunIOType


def test_url_with_spaces():
    url_with_spaces = "http://example.com/some file.png"
    expected_url = "http://example.com/some%20file.png"
    run_input = RunInput(file_url=url_with_spaces, type=RunIOType.file)
    assert (
        run_input.file_url == expected_url
    ), "URL with spaces should be encoded correctly"


def test_url_without_spaces():
    url_without_spaces = "http://example.com/file.png"
    run_input = RunInput(file_url=url_without_spaces, type=RunIOType.file)
    assert (
        run_input.file_url == url_without_spaces
    ), "URL without spaces should not be altered"


def test_none_url():
    run_input = RunInput(file_url=None, type=RunIOType.file)
    assert run_input.file_url is None, "None URL should remain None"
