import argparse
import sys

import requests

from pipeline import configuration
from pipeline.util.logging import _print


def main(args) -> int:
    base_parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Create or run pipelines locally or in the cloud!",
        add_help=True,
    )
    base_parser.add_argument(
        "-v",
        "--verbose",
        required=False,
        help="Verbose logging",
        default=False,
        type=bool,
    )

    command_parser = base_parser.add_subparsers(dest="command")
    login_parser = command_parser.add_parser(
        "login", description="Authenticate with a remote compute service"
    )

    login_parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        help="remote URL for auth (default=https://api.pipeline.ai)",
        default="https://api.pipeline.ai",
    )
    login_parser.add_argument(
        "-t", "--token", type=str, required=True, help="API token for remote"
    )

    args: argparse.Namespace = base_parser.parse_args(args)

    if args.command == "login":
        url = f"{args.url}/v2/users/me"

        headers = {"Authorization": f"Bearer {args.token}"}
        try:
            response = requests.request("GET", url, headers=headers)

            if response.status_code == 200:
                configuration.remote_auth[args.url] = args.token
                configuration._save_auth()
                _print(f"Successfully authenticated with {args.url}")
                return 0
        except requests.exceptions.ConnectionError:
            _print(f"Couldn't connect to host {url}", level="ERROR")
            return 1

        if args.verbose:
            print(response.status_code)
            print(response.text)
        _print(f"Couldn't authenticate with {args.url}", level="ERROR")
        return 1
    else:
        base_parser.print_help()
        return 0


def _run() -> int:
    sys.exit(main(sys.argv[1:]))
