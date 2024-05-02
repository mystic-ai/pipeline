from pipeline.cloud.schemas.runs import RunInput, RunIOType


def test_url_with_spaces():
    url_with_spaces = "http://example.com/some file.png"
    expected_url = "http://example.com/some%20file.png"
    # First instance of RunInput
    run_input1 = RunInput(file_url=url_with_spaces, type=RunIOType.file)
    assert (
        run_input1.file_url == expected_url
    ), "URL with spaces should be encoded correctly in the first instance"

    # Second instance of RunInput using the file_url from the first instance
    run_input2 = RunInput(file_url=run_input1.file_url, type=RunIOType.file)
    assert (
        run_input2.file_url == expected_url
    ), "URL with spaces should be encoded correctly in the second instance"

def test_url_without_spaces():
    url_without_spaces = "http://example.com/file.png"
    run_input = RunInput(file_url=url_without_spaces, type=RunIOType.file)
    assert (
        run_input.file_url == url_without_spaces
    ), "URL without spaces should not be altered"


def test_none_url():
    run_input = RunInput(file_url=None, type=RunIOType.file)
    assert run_input.file_url is None, "None URL should remain None"
