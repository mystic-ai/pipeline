import argparse
import sys
import traceback
from typing import List, Optional

from pipeline.console.commands import (
    cluster_parser,
    container_parser,
    create_parser,
    delete_parser,
    edit_parser,
    get_parser,
    logs_parser,
)


def construct_cli() -> argparse.ArgumentParser:
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

    create_parser(command_parser)
    get_parser(command_parser)
    edit_parser(command_parser)
    delete_parser(command_parser)
    cluster_parser(command_parser)
    logs_parser(command_parser)
    container_parser(command_parser)

    return base_parser


def execute_cli(
    parser: argparse.ArgumentParser,
    args: Optional[List[str]] = None,
) -> None:
    if args is None:
        args = sys.argv[1:]

    parsed_args = parser.parse_args(args=args)

    if (selected_func := getattr(parsed_args, "func", None)) is not None:
        selected_func(parsed_args)

    if parsed_args.command is None:
        parser.print_help()
        return


def _run() -> int:
    try:
        parser = construct_cli()
        execute_cli(parser)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    sys.exit(0)
