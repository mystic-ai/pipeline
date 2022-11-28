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
        "login", help="Authenticate with a remote compute service"
    )

    login_parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        help="Remote URL for auth",
        default="api.pipeline.ai",
    )
    login_parser.add_argument(
        "-t", "--token", type=str, required=True, help="API token for remote"
    )

    args: argparse.Namespace = base_parser.parse_args(args)

    if args.command == "login":
        url = f"https://{args.url}/v2/users/me"

        headers = {"Authorization": f"Bearer {args.token}"}
        try:
            response = requests.request("GET", url, headers=headers)

            if response.status_code == 200:
                configuration.remote_auth[args.url] = args.token
                configuration._save_auth()
                _print(f"Successfully authenticated with {args.url}")
                return

        except requests.exceptions.ConnectionError:
            _print(f"Couldn't connect to host {url}", level="ERROR")
        if args.verbose:
            print(response.status_code)
            print(response.text)
        _print(f"Couldn't authenticate with {args.url}", level="ERROR")
    else:
        base_parser.print_help()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
