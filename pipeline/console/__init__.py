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
        valid_token = PipelineCloud._validate_token(args.token, args.url)

        if valid_token:
            configuration.remote_auth[args.url] = args.token
            configuration._save_auth()
            _print(f"Successfully authenticated with {args.url}")
            return 0

        _print(f"Couldn't authenticate with {args.url}", level="ERROR")
        return 1
    else:
        base_parser.print_help()
        return 0


def _run() -> int:
    sys.exit(main())
