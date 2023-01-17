import pytest

from pipeline import configuration
from pipeline.console import main as cli_main
from pipeline.console.environments import _get_environment
from pipeline.schemas.environment import EnvironmentGet


def _set_testing_remote_compute_service(url, token):
    cli_main(["remote", "login", "-u", url, "-t", token])
    cli_main(["remote", "set", url])
    configuration.DEFAULT_REMOTE = url


@pytest.mark.usefixtures("top_api_server")
def test_cli_environments_get(
    environment_get: EnvironmentGet,
    url: str,
    token: str,
):
    _set_testing_remote_compute_service(url, token)
    assert environment_get == _get_environment(environment_get.name)
    assert environment_get == _get_environment(environment_get.id)
