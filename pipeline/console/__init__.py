import argparse
import sys
from typing import List, Optional

from pipeline.console.remote import remote as remote_command
from pipeline.console.runs import runs as runs_command


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

    ##########
    # pipeline runs
    ##########

    runs_parser = command_parser.add_parser(
        "runs",
        description="Manage pipeline runs",
        help="Manage pipeline runs",
    )
    runs_sub_parser = runs_parser.add_subparsers(dest="sub-command")

    ##########
    # pipeline runs list
    ##########

    runs_sub_parser.add_parser(
        "list",
        aliases=["ls"],
        description="List the currently executing runs",
        help="List the currently executing runs",
    )
    ##########
    # pipeline runs get
    ##########

    runs_get_parser = runs_sub_parser.add_parser(
        "get",
        description="Get run information from a remote compute service",
        help="Get run information from a remote compute service",
    )

    runs_get_parser.add_argument("run_id", help="The run id")

    runs_get_parser.add_argument(
        "-r",
        "--result",
        action="store_true",
        help="Get the run result",
    )

    args: argparse.Namespace = base_parser.parse_args(args)
    command = getattr(args, "command", None)

    if command == "remote":
        if (code := remote_command(args)) is None:
            remote_parser.print_help()
            return 1
    elif command == "runs":
        if (code := runs_command(args)) is None:
            runs_parser.print_help()
            return 1
    else:
        base_parser.print_help()
        return 0

    return code


def _run() -> int:
    sys.exit(main())
