from argparse import ArgumentParser, Namespace, _SubParsersAction
from datetime import datetime

from tabulate import tabulate

from pipeline.cloud import http
from pipeline.cloud.schemas.pagination import (
    Paginated,
    get_default_pagination,
    to_page_position,
)
from pipeline.util.logging import _print


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
        ],
        tablefmt="psql",
    )
    print(table)
    print(f"\nPage {page_position['current']} of {page_position['total']}\n")


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
