import pytest
from _pytest.capture import CaptureFixture
from httpx import HTTPStatusError

from pipeline import configuration
from pipeline.console import main as cli_main
from pipeline.console.tags import (
    _delete_tag,
    _get_tag,
    _list_tags,
    _update_or_create_tag,
)
from pipeline.schemas.pagination import Paginated
from pipeline.schemas.pipeline import PipelineTagGet


def _set_testing_remote_compute_service(url, token):
    cli_main(["remote", "login", "-u", url, "-t", token])
    cli_main(["remote", "set", url])
    configuration.DEFAULT_REMOTE = url


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


##########
# pipeline tags
##########


@pytest.mark.usefixtures("top_api_server")
def test_tags_create(
    url: str,
    token: str,
    tag_get: PipelineTagGet,
    tag_get_2: PipelineTagGet,
):

    _set_testing_remote_compute_service(url=url, token=token)
    with pytest.raises(SystemExit):
        cli_main(["tags", "create", "bad_tag", "pipeline_id"])
    assert cli_main(["tags", "create", tag_get_2.pipeline_id, tag_get.name]) == 0
    assert cli_main(["tags", "create", tag_get_2.name, tag_get.name]) == 0

    create_output_by_pipeline_id = _update_or_create_tag(
        tag_get_2.pipeline_id, tag_get.name, "create"
    )
    create_output_by_tag_name = _update_or_create_tag(
        tag_get_2.name, tag_get.name, "create"
    )

    assert create_output_by_pipeline_id == tag_get
    assert create_output_by_tag_name == tag_get


@pytest.mark.usefixtures("top_api_server")
def test_tags_update(
    url: str,
    token: str,
    tag_get: PipelineTagGet,
    tag_get_3: PipelineTagGet,
    tag_get_patched: PipelineTagGet,
):
    _set_testing_remote_compute_service(url=url, token=token)
    with pytest.raises(SystemExit):
        cli_main(["tags", "update", "pipeline_id", "bad_tag"])

    # Attempting to update a missing tag should raise a 404 in the `_get_tag` function
    with pytest.raises(HTTPStatusError):
        cli_main(
            [
                "tags",
                "update",
                "pipeline_id",
                "missing:tag",
            ]
        )

    update_by_pipeline_id = _update_or_create_tag(
        tag_get_3.pipeline_id, tag_get.name, "update"
    )
    update_by_tag_name = _update_or_create_tag(tag_get_3.name, tag_get.name, "update")

    assert update_by_pipeline_id == tag_get_patched
    assert update_by_tag_name == tag_get_patched


@pytest.mark.usefixtures("top_api_server")
def test_tags_get(
    url: str,
    token: str,
    tag_get: PipelineTagGet,
):
    _set_testing_remote_compute_service(url=url, token=token)
    with pytest.raises(SystemExit):
        cli_main(["tags", "get", "bad_tag"])

    # Attempting to update a missing tag should raise a 404 in the `_get_tag` function
    with pytest.raises(HTTPStatusError):
        cli_main(["tags", "get", "missing:tag"])

    get_tag_by_name = _get_tag(tag_get.name)

    assert get_tag_by_name == tag_get


@pytest.mark.usefixtures("top_api_server")
@pytest.mark.parametrize("option", ("list", "ls"))
def test_tags_list(
    option: str,
    url: str,
    token: str,
    tags_list: Paginated[PipelineTagGet],
):
    _set_testing_remote_compute_service(url=url, token=token)

    assert cli_main(["tags", option, "-p", "pipeline_id", "-l", "5", "-s", "1"]) == 0

    list_tags = _list_tags(skip=1, limit=5, pipeline_id="pipeline_id")

    assert list_tags == tags_list


@pytest.mark.usefixtures("top_api_server")
@pytest.mark.parametrize("option", ("delete", "rm"))
def test_tags_delete(option: str, url: str, token: str, tag_get: PipelineTagGet):
    _set_testing_remote_compute_service(url=url, token=token)

    with pytest.raises(SystemExit):
        cli_main(["tags", option, "bad_tag"])

    # Attempting to delete a missing tag should raise a 404
    with pytest.raises(HTTPStatusError):
        cli_main(["tags", option, "missing:tag"])

    assert _delete_tag(tag_get.name) is None
