import json
from argparse import ArgumentParser, Namespace, _SubParsersAction
from datetime import datetime
from enum import Enum

from tabulate import tabulate

from pipeline.cloud import http
from pipeline.cloud.schemas.pagination import (
    Paginated,
    get_default_pagination,
    to_page_position,
)
from pipeline.util.logging import _print


class ScalingConfigType(str, Enum):
    windows = "windows"


def _create_scaling_config(namespace: Namespace) -> None:
    name = getattr(namespace, "name")
    type_ = getattr(namespace, "type")
    args_ = getattr(namespace, "args")
    min_nodes = getattr(namespace, "min_nodes")
    max_nodes = getattr(namespace, "max_nodes")
    # Annoyingly, setting default values above did not seem to work
    payload = {}
    payload["name"] = name
    payload["type"] = type_ or "windows"
    payload["args"] = args_ or {}
    payload["minimum_nodes"] = min_nodes or 1
    payload["maximum_nodes"] = max_nodes or 100

    result = http.post("/v4/scaling-configs", json_data=payload)

    scaling_config = result.json()["name"]

    _print(f"Created scaling configuration {scaling_config}")


def _get_scaling_config(args: Namespace) -> None:
    _print("Getting scaling configurations")

    params = dict()
    pagination = get_default_pagination()
    if name := getattr(args, "name", None):
        params["name"] = name
    if skip := getattr(args, "skip", None):
        pagination.skip = skip
    if limit := getattr(args, "limit", None):
        pagination.limit = limit
    paginated_scaling_configs: Paginated[dict] = http.get(
        "/v4/scaling-configs",
        params=dict(**params, **pagination.dict()),
    ).json()

    scaling_configs = [
        [
            scaling["id"],
            scaling["name"],
            datetime.fromtimestamp(scaling.get("created_at"))
            if "created_at" in scaling
            else "N/A",
            scaling["type"],
            scaling["args"],
        ]
        for scaling in paginated_scaling_configs["data"]
    ]

    page_position = to_page_position(
        paginated_scaling_configs["skip"],
        paginated_scaling_configs["limit"],
        paginated_scaling_configs["total"],
    )

    table = tabulate(
        scaling_configs,
        headers=[
            "ID",
            "Name",
            "Created",
            "Type",
            "Args",
        ],
        tablefmt="psql",
    )
    print(table)
    print(f"\nPage {page_position['current']} of {page_position['total']}\n")


def _edit_scaling_config(args: Namespace) -> None:
    # If you add new arg, check whether it should be add to
    # the `if all(...)` below
    name = getattr(args, "name")
    type_ = getattr(args, "type", None)
    args_ = getattr(args, "args", None)
    min_nodes = getattr(args, "min_nodes")
    max_nodes = getattr(args, "max_nodes")

    if all(arg is None for arg in (type_, args_, min_nodes, max_nodes)):
        _print("Nothing to edit.", level="ERROR")
        return

    payload = {}
    if type_ is not None:
        payload["type"] = type_
    if args_ is not None:
        payload["args"] = args_
    if min_nodes is not None:
        payload["minimum_nodes"] = min_nodes
    if max_nodes is not None:
        payload["maximum_nodes"] = max_nodes

    response = http.patch(
        f"/v4/scaling-configs/{name}",
        payload,
    )
    _print(response.json())

    _print("Scaling configuration edited!")


def _delete_scaling_config(args: Namespace) -> None:
    name = getattr(args, "name")

    http.delete(
        f"/v4/scaling-configs/{name}",
    )

    _print("Scaling configuration deleted!")


def get_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    get_parser = command_parser.add_parser(
        "scalings",
        aliases=["scaling"],
        help="Get scaling config information.",
    )

    get_parser.set_defaults(func=_get_scaling_config)

    # get by name
    get_parser.add_argument(
        "--name",
        "-n",
        help="Scaling config name.",
        type=str,
    )
    get_parser.add_argument(
        "--skip",
        "-s",
        help="Number of scaling configs to skip in paginated set.",
        type=int,
    )
    get_parser.add_argument(
        "--limit",
        "-l",
        help="Total number of scaling configs to fetch in paginated set.",
        type=int,
    )


def create_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    create_parser = command_parser.add_parser(
        "scalings",
        aliases=["scaling"],
        help="Create a new scaling configuration.",
    )

    create_parser.set_defaults(func=_create_scaling_config)

    create_parser.add_argument(
        "name",
        type=str,
        help="Scaling configuration to create.",
    )
    create_parser.add_argument(
        "--type",
        "-t",
        help="The type of the scaling configuration",
        type=ScalingConfigType,
    )
    create_parser.add_argument(
        "--args",
        "-a",
        help="The arguments of the scaling configuration",
        type=json.loads,
    )
    create_parser.add_argument(
        "--min-nodes",
        help="The minimum number of nodes for the scaling configuration",
        type=int,
    )
    create_parser.add_argument(
        "--max-nodes",
        help="The maximum number of nodes for the scaling configuration",
        type=int,
    )


def edit_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    edit_parser = command_parser.add_parser(
        "scalings",
        aliases=["scaling"],
        help="Edit scaling config information.",
    )

    edit_parser.set_defaults(func=_edit_scaling_config)

    # Requires name param to edit
    edit_parser.add_argument(
        "name",
        help="The name of the scaling configuration to edit.",
        type=str,
    )
    edit_parser.add_argument(
        "--type",
        "-t",
        help="The type of the scaling configuration",
        type=ScalingConfigType,
    )
    edit_parser.add_argument(
        "--args",
        "-a",
        help="The arguments of the scaling configuration",
        type=json.loads,
    )
    edit_parser.add_argument(
        "--min-nodes",
        help="The minimum number of nodes of the scaling configuration",
        type=int,
    )
    edit_parser.add_argument(
        "--max-nodes",
        help="The maximum number of nodes of the scaling configuration",
        type=int,
    )


def delete_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    delete_parser = command_parser.add_parser(
        "scalings",
        aliases=["scaling"],
        help="Delete a scaling configuration.",
    )

    delete_parser.set_defaults(func=_delete_scaling_config)

    delete_parser.add_argument(
        "name",
        help="Name of the scaling configuration to delete.",
    )
