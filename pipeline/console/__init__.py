import argparse
import sys
from typing import List, Optional

from pipeline import configuration
from pipeline.api import PipelineCloud
from pipeline.util.logging import _print


def main(args: Optional[List[str]] = None) -> int:
    base_parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Create or run pipelines locally or in the cloud!",
        add_help=True,
    )
    base_parser.add_argument(
        "-v",
        "--verbose",
        help="Verbose logging",
        action="store_true",
    )

    command_parser = base_parser.add_subparsers(dest="command")
    ##########
    # pipeline remote
    ##########

    remote_parser = command_parser.add_parser(
        "remote",
        description="Manage remote compute services",
        help="Manage remote compute services",
    )
    remote_sub_parser = remote_parser.add_subparsers(dest="sub-command")

    ##########
    # pipeline remote login
    ##########
    remote_login_parser = remote_sub_parser.add_parser(
        "login",
        description="Authenticate with a remote compute service",
        help="Authenticate with a remote compute service",
    )

    remote_login_parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        help="Remote URL for auth (default=https://api.pipeline.ai)",
        # description="Remote URL for auth (default=https://api.pipeline.ai)",
        default="https://api.pipeline.ai",
    )
    remote_login_parser.add_argument(
        "-t",
        "--token",
        type=str,
        required=True,
        help="API token for remote",
        # description="API token for remote",
    )

    ##########
    # pipeline remote set
    ##########

    remote_set_parser = remote_sub_parser.add_parser(
        "set",
        description="Set the currently used remote compute service URL",
        help="Set the currently used remote compute service URL",
    )
    remote_set_parser.add_argument("url", help="The remote URL")

    ##########
    # pipeline remote list
    ##########

    remote_sub_parser.add_parser(
        "list",
        aliases=["ls"],
        description="List the authenticated remote compute services",
        help="List the authenticated remote compute services",
    )

    args: argparse.Namespace = base_parser.parse_args(args)
    command = getattr(args, "command", None)
    sub_command = getattr(args, "sub-command", None)

    if command == "remote":
        if sub_command == "set":
            default_url = args.url
            configuration.config["DEFAULT_REMOTE"] = default_url
            configuration._save_config()
            _print(
                f"Set new default remote to '{configuration.config['DEFAULT_REMOTE']}'"
            )
            return 0
        elif sub_command in ["list", "ls"]:
            remotes = [
                f"{_remote} (active)"
                if _remote == configuration.DEFAULT_REMOTE
                else f"{_remote}"
                for _remote in configuration.remote_auth.keys()
            ]
            _print("Authenticated remotes:")
            [print(_remote) for _remote in remotes]
            return 0
        elif sub_command == "login":
            valid_token = PipelineCloud._validate_token(args.token, args.url)

            if valid_token:
                configuration.remote_auth[args.url] = args.token
                configuration._save_auth()
                _print(f"Successfully authenticated with {args.url}")
                return 0

            _print(f"Couldn't authenticate with {args.url}", level="ERROR")
            return 1
        else:
            return 0
    else:
        base_parser.print_help()
        return 0


def _run() -> int:
    sys.exit(main())
