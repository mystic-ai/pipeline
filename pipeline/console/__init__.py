import argparse
import json
import sys
from typing import List, Optional

from tabulate import tabulate

from pipeline import configuration
from pipeline.api import PipelineCloud
from pipeline.schemas.file import FileGet
from pipeline.schemas.pagination import Paginated
from pipeline.schemas.run import RunGet, RunState
from pipeline.util import hex_to_python_object
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
            remote_parser.print_help()
            return 1
    elif command == "runs":

        remote_service = PipelineCloud(verbose=False)
        remote_service.authenticate()

        if sub_command in ["list", "ls"]:
            raw_result = remote_service.get_runs()

            schema = Paginated[RunGet].parse_obj(raw_result)

            runs = schema.data

            terminal_run_states = [
                RunState.FAILED,
                RunState.COMPLETE,
            ]

            run_data = [
                [
                    _run.id,
                    _run.created_at.strftime("%d-%m-%Y %H:%M:%S"),
                    "executing",
                    _run.runnable.id,
                ]
                for _run in runs
                if _run.run_state not in terminal_run_states
            ]
            table = tabulate(
                run_data,
                headers=[
                    "ID",
                    "Created at",
                    "State",
                    "Pipeline",
                ],
                tablefmt="outline",
            )
            print(table)
            return 0
        elif sub_command == "get":
            run_id = args.run_id

            result = remote_service._get(f"/v2/runs/{run_id}")
            if args.result:
                result = RunGet.parse_obj(result)
                if result.result_preview is not None:
                    print(json.dumps(result.result_preview))
                else:
                    file_schema_raw = remote_service._get(
                        f"/v2/files/{result.result.id}?return_data=true"
                    )

                    file_schema = FileGet.parse_obj(file_schema_raw)
                    raw_result = hex_to_python_object(file_schema.data)
                    print(json.dumps(raw_result))

                return 0
            else:
                print(json.dumps(result))
                return 0

        else:
            runs_parser.print_help()
            return 1

    else:
        base_parser.print_help()
        return 0


def _run() -> int:
    sys.exit(main())
