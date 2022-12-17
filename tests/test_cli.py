import pytest
from _pytest.capture import CaptureFixture

from pipeline import configuration
from pipeline.console import main as cli_main
from pipeline.schemas.pipeline import PipelineTagGet


@pytest.mark.parametrize("option", ("-h", "--help"))
def test_help(capsys: CaptureFixture, option):
    try:
        cli_main([option])
    except SystemExit as system_exit:
        assert system_exit.code == 0

    output: str = capsys.readouterr().out
    assert output.startswith("usage: pipeline")


def test_login(url, top_api_server, token):
    response_code = cli_main(["remote", "login", "-u", url, "-t", token])
    assert response_code == 0


def test_login_fail(url, top_api_server_bad_token, bad_token):
    response_code = cli_main(["remote", "login", "-u", url, "-t", bad_token])
    assert response_code == 1


@pytest.mark.parametrize("sub_command", ("list", "ls"))
def test_remote_list(capsys: CaptureFixture, sub_command):
    configuration.remote_auth = dict(test_url="test_token", test_url2="test_token2")
    configuration._save_auth()
    configuration.config["DEFAULT_REMOTE"] = "test_url"
    configuration._save_config()

    configuration._load_config()
    configuration._load_auth()

    response_code = cli_main(["remote", sub_command])
    assert response_code == 0
    output: str = capsys.readouterr().out
    remotes = output.splitlines()
    assert remotes[1:] == ["test_url (active)", "test_url2"]


@pytest.mark.parametrize("option", ("-v", "--verbose"))
def test_verbose(
    capsys: CaptureFixture,
    option,
):
    response_code = cli_main([option])
    assert response_code == 0

    output: str = capsys.readouterr().out
    assert output.startswith("usage: pipeline")


@pytest.mark.parametrize("option", ("list", "ls"))
def test_runs_list(url, token, option, capsys, top_api_server):
    cli_main(["remote", "login", "-u", url, "-t", token])
    cli_main(["remote", "set", url])
    configuration.DEFAULT_REMOTE = url

    response = cli_main(["runs", option])

    output: str = capsys.readouterr().out
    runs = output.splitlines()
    assert response == 0
    assert "| run_test_2 | 01-01-2000 00:00:00 | executing | test_function_id |" in runs


def test_runs_get(url, token, capsys, run_get, top_api_server):
    cli_main(["remote", "login", "-u", url, "-t", token])
    cli_main(["remote", "set", url])
    configuration.DEFAULT_REMOTE = url

    response = cli_main(["runs", "get", run_get.id])

    output: str = str(capsys.readouterr().out)
    output_lines = output.splitlines()
    assert response == 0
    assert output_lines[2] == run_get.json()

    response = cli_main(["runs", "get", run_get.id, "-r"])

    output: str = str(capsys.readouterr().out)
    output_lines = output.splitlines()
    assert response == 0
    assert output == '{"test": "hello"}\n'


@pytest.mark.usefixtures("api_response")
def test_tags_create(
    url: str,
    token: str,
    tag_get: PipelineTagGet,
    capsys: CaptureFixture,
):
    response = cli_main(["tags", "create", tag_get.name, "pipeline_id"])
    assert response == 0
