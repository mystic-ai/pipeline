from argparse import ArgumentParser, Namespace, _SubParsersAction

from pipeline import PipelineCloud, current_configuration
from pipeline.exceptions.MissingActiveToken import MissingActiveToken
from pipeline.util.logging import _print


def login_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    login_parser = command_parser.add_parser(
        "login", help="Login with a compute cluster."
    )
    login_parser.set_defaults(func=_login)

    login_parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        help="Remote URL for auth (default=https://api.pipeline.ai)",
        default="https://api.pipeline.ai",
    )

    login_parser.add_argument(
        "alias",
        help="Name to use for cluster.",
    )

    login_parser.add_argument(
        "token",
        help="API token for cluster",
    )

    login_parser.add_argument(
        "-a",
        "--active",
        help="Set as the current active remote",
        action="store_true",
    )


def use_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    use_parser = command_parser.add_parser(
        "use",
        help="Use a compute cluster.",
    )
    use_parser.set_defaults(func=_use)

    use_parser.add_argument(
        "alias",
        help="Name of the cluster to use.",
    )


def remove_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    ...


def get_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    get_parser = command_parser.add_parser(
        "get",
        help="Get information about the current remotes.",
    )
    get_parser.set_defaults(func=_get)


def _login(namespace: Namespace) -> None:
    alias = getattr(namespace, "alias")
    url = getattr(namespace, "url")
    token = getattr(namespace, "token")
    active = getattr(namespace, "token", False)

    try:
        PipelineCloud(token=token, url=url, verbose=False)
    except MissingActiveToken:
        _print(f"Couldn't authenticate with {url}", level="ERROR")
        return

    current_configuration.add_remote(
        alias=alias,
        url=url,
        token=token,
    )

    if active:
        _print(f"Setting new remote as active '{alias}'")
        current_configuration.set_active_remote(alias)

    _print(f"Successfully authenticated with {alias}")


def _use(namespace: Namespace) -> None:
    alias = getattr(namespace, "alias")
    if not any([remote.alias == alias for remote in current_configuration.remotes]):
        _print(f"Remote '{alias}' not found", level="ERROR")
        return

    current_configuration.set_active_remote(alias)
    _print(f"Set new default remote to '{alias}'")


def _get(namespace: Namespace) -> None:
    remotes = [
        f"{_remote} (active)" if _remote.active else f"{_remote}"
        for _remote in current_configuration.remotes
    ]
    _print("Authenticated remotes:")
    [print(_remote) for _remote in remotes]
