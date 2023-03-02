import argparse
import sys
from typing import List, Optional

from pipeline.console.environments import environments as environments_command
from pipeline.console.remote import remote as remote_command
from pipeline.console.runs import runs as runs_command
from pipeline.console.tags import tags as tags_command


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

    runs_get_parser.add_argument(
        "run_id",
        help="The run id",
    )

    runs_get_parser.add_argument(
        "-r",
        "--result",
        action="store_true",
        help="Get the run result",
    )

    ##########
    # pipeline tags
    ##########

    tags_parser = command_parser.add_parser(
        "tags",
        description="Manage pipeline tags",
        help="Manage pipeline tags",
    )

    tags_sub_parser = tags_parser.add_subparsers(dest="sub-command")

    ##########
    # pipeline tags create
    ##########

    tags_set_parser = tags_sub_parser.add_parser(
        "create",
        help="Create a tag TARGET that points to SOURCE",
    )

    tags_set_parser.add_argument(
        "source",
        help="The source pipeline:tag or pipeline_id",
    )
    tags_set_parser.add_argument(
        "target",
        help="The target pipeline:tag",
    )

    ##########
    # pipeline tags update
    ##########

    tags_update_parser = tags_sub_parser.add_parser(
        "update",
        help="Update a tag TARGET to point to a new SOURCE",
    )
    tags_update_parser.add_argument(
        "source",
        help="The source pipeline:tag or pipeline_id",
    )
    tags_update_parser.add_argument(
        "target",
        help="The target pipeline:tag",
    )

    ##########
    # pipeline tags list
    ##########

    tags_list_parser = tags_sub_parser.add_parser(
        "list",
        aliases=["ls"],
        help="List tags on the remote compute service",
    )

    tags_list_parser.add_argument(
        "-p",
        "--pipeline-id",
        type=str,
        help="Filter by target pipeline id",
    )
    tags_list_parser.add_argument(
        "-l",
        "--limit",
        required=False,
        help="Number of tags to get",
        default=20,
        type=int,
    )
    tags_list_parser.add_argument(
        "-s",
        "--skip",
        required=False,
        help="Number of tags to skip for pagination",
        default=0,
        type=int,
    )

    ##########
    # pipeline tags delete
    ##########

    tags_delete_parser = tags_sub_parser.add_parser(
        "delete",
        aliases=["rm"],
        help="Delete a pipeline tag (does not delete the pipeline)",
    )

    tags_delete_parser.add_argument(
        "pipeline_tag", help="The pipeline tag or tag_id to delete"
    )

    ##########
    # pipeline tags get
    ##########

    tags_get_parser = tags_sub_parser.add_parser(
        "get",
        help="Get tag information",
    )

    tags_get_parser.add_argument(
        "pipeline_tag", help="The pipeline tag or tag_id to get"
    )

    ##########
    # pipeline environments
    ##########

    environments_parser = command_parser.add_parser(
        "environments",
        description="Manage environments",
        help="Manage environments",
    )

    environments_sub_parser = environments_parser.add_subparsers(dest="sub-command")

    ##########
    # pipeline environments create
    ##########

    environments_create_parser = environments_sub_parser.add_parser(
        "create",
        help="Create a new python environment",
    )

    environments_create_parser.add_argument(
        "name",
        help="The environment name",
    )

    environments_create_parser.add_argument(
        "-r",
        "--requirements",
        type=str,
        help="Create environment from requirements file",
        required=False,
    )
    environments_create_parser.add_argument(
        "-l",
        "--from-local",
        action="store_true",
        help="Create environment using all locally installed packages",
        required=False,
        default=False,
    )

    ##########
    # pipeline environments get
    ##########

    environments_get_parser = environments_sub_parser.add_parser(
        "get",
        help="Get environment information",
    )

    environments_get_parser.add_argument(
        "name_or_id", help="The environment name or id to get"
    )

    environments_get_parser.add_argument(
        "-n",
        help="Get the environment by name",
        required=False,
        action="store_true",
        default=False,
    )
    ##########
    # pipeline environments list
    ##########
    environments_list_parser = environments_sub_parser.add_parser(
        "list",
        aliases=["ls"],
        help="List environments",
    )
    environments_list_parser.add_argument(
        "-l",
        "--limit",
        required=False,
        help="Number of environments to list",
        default=20,
        type=int,
    )

    environments_list_parser.add_argument(
        "-s",
        "--skip",
        required=False,
        help="Number of environments to skip for pagination",
        default=0,
        type=int,
    )

    ##########
    # pipeline environments delete
    ##########
    environments_delete_parser = environments_sub_parser.add_parser(
        "delete",
        aliases=["rm"],
        help="Delete environment",
    )

    environments_delete_parser.add_argument(
        "name_or_id", help="The environment name or id to delete"
    )

    environments_delete_parser.add_argument(
        "-n",
        help="Delete the environment by name",
        required=False,
        action="store_true",
        default=False,
    )

    ##########
    # pipeline environments update
    ##########
    environments_update_parser = environments_sub_parser.add_parser(
        "update",
        help="Update environment",
    )

    environments_update_parser.add_argument(
        "name_or_id", help="The environment name or id to update"
    )
    environments_update_parser.add_argument(
        "-n",
        help="Update the environment by name",
        required=False,
        action="store_true",
        default=False,
    )

    environments_update_sub_command = environments_update_parser.add_subparsers(
        dest="environments-update-sub-command"
    )

    environments_update_sub_command.add_parser("lock", help="Lock environment")
    environments_update_sub_command.add_parser("unlock", help="Unlock environment")

    environments_update_add_command = environments_update_sub_command.add_parser(
        "add", help="Add packages"
    )

    environments_update_remove_command = environments_update_sub_command.add_parser(
        "remove", help="Remove packages"
    )

    environments_update_add_command.add_argument(
        "packages",
        metavar="package",
        type=str,
        nargs="+",
    )
    environments_update_remove_command.add_argument(
        "packages",
        metavar="package",
        type=str,
        nargs="+",
    )
    ##########

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
    elif command == "tags":
        if (code := tags_command(args)) is None:
            tags_parser.print_help()
            return 1
    elif command == "environments":
        if (code := environments_command(args)) is None:
            if getattr(args, "sub-command") == "update":
                environments_update_parser.print_help()
            else:
                environments_parser.print_help()
            return 1
    else:
        base_parser.print_help()
        return 0

    return code


def _run() -> int:
    sys.exit(main())
