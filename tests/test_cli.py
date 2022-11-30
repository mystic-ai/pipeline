import pytest

from pipeline.console import main as cli_main


@pytest.mark.parametrize("option", ("-h", "--help"))
def test_help(capsys, option):
    try:
        cli_main([option])
    except SystemExit as system_exit:
        assert system_exit.code == 0

    output: str = capsys.readouterr().out
    assert output.startswith("usage: pipeline")


@pytest.mark.usefixtures("api_response")
def test_login(url, token):
    response_code = cli_main(["login", "-u", url, "-t", token])
    assert response_code == 0


@pytest.mark.usefixtures("api_response")
def test_login_fail(url, bad_token):
    response_code = cli_main(["login", "-u", url, "-t", bad_token])
    assert response_code == 1


@pytest.mark.parametrize("option", ("-v", "--verbose"))
def test_verbose(capsys, option):
    response_code = cli_main([option])
    assert response_code == 0

    output: str = capsys.readouterr().out
    assert output.startswith("usage: pipeline")
