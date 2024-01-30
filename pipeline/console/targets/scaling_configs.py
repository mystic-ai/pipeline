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
    name = getattr(args, "name")
    type_ = getattr(args, "type", None)
    args_ = getattr(args, "args", None)

    # patch_schema = pipelines_schema.PipelinePatch(
    #     minimum_cache_number=cache_number,
    #     gpu_memory_min=gpu_memory,
    # )

    if type_ is None and args_ is None:
        _print("Nothing to edit.", level="ERROR")
        return

    payload = {}
    if type_ is not None:
        payload["type"] = type_
    if args_ is not None:
        payload["args"] = args_
    http.patch(
        f"/v4/scaling-configs/{name}",
        payload,
    )

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
