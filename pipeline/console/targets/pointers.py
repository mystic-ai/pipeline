import json
import re
from argparse import ArgumentParser, Namespace, _SubParsersAction

from tabulate import tabulate

from pipeline.cloud import http
from pipeline.cloud.schemas import pointers as pointers_schema
from pipeline.cloud.schemas.pagination import (
    Paginated,
    get_default_pagination,
    to_page_position,
)
from pipeline.util.logging import _print

VALID_TAG_NAME = re.compile(
    r"^[a-z0-9][a-z0-9-._/]*[a-z0-9]:[0-9A-Za-z_][0-9A-Za-z-_.]{0,127}$"
)


def create_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    create_parser = command_parser.add_parser(
        "pointers",
        aliases=["pointer", "ptr"],
        help="Create a new pointer.",
    )

    create_parser.add_argument(
        "new_pointer",
        type=str,
        help="Pointer to create.",
    )

    create_parser.add_argument(
        "pipeline_id_or_pointer",
        type=str,
        help="Pipeline id or pointer to create a pointer to.",
    )

    create_parser.add_argument(
        "--locked",
        action="store_true",
        help="Lock the pointer.",
    )

    create_parser.set_defaults(func=_create_pointer)


def edit_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    edit_parser = command_parser.add_parser(
        "pointers",
        aliases=["pointer", "ptr"],
        help="Edit a pointer.",
    )

    edit_parser.add_argument(
        "pointer",
        type=str,
        help="Pointer to edit.",
    )

    edit_parser.add_argument(
        "--source",
        "-s",
        type=str,
        help="Pipeline id or pointer to create a pointer to.",
        default=None,
    )

    edit_parser.add_argument(
        "--locked",
        action="store_true",
        help="Lock the pointer.",
    )
    edit_parser.add_argument(
        "--unlocked",
        action="store_true",
        help="Lock the pointer.",
    )
    edit_parser.set_defaults(func=_edit_pointer)


def get_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    get_parser = command_parser.add_parser(
        "pointers",
        aliases=["pointer", "ptr"],
        help="Get pointer information.",
    )

    get_parser.set_defaults(func=_get_pointer)

    get_parser.add_argument(
        "--name",
        "-n",
        type=str,
        help="Pipeline name.",
    )

    get_parser.add_argument(
        "--show-deleted",
        "-d",
        action="store_true",
        help="Show pointers to deleted pipelines.",
    )
    get_parser.add_argument(
        "--skip",
        "-s",
        help="Number of pointers to skip in paginated set.",
        type=int,
    )
    get_parser.add_argument(
        "--limit",
        "-l",
        help="Total number of pointer to fetch in paginated set.",
        type=int,
    )


def delete_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    delete_parser = command_parser.add_parser(
        "pointers",
        aliases=["pointer", "ptr"],
        help="Delete a pointer.",
    )

    delete_parser.add_argument(
        "pointer",
        type=str,
        help="Pointer to delete.",
    )

    delete_parser.set_defaults(func=_delete_pointer)


def _get_pointer(namespace: Namespace) -> None:
    _print("Getting pointers")

    pipeline_name = getattr(namespace, "name", None)
    show_deleted = getattr(namespace, "show_deleted", False)
    query_params = dict()
    pagination = get_default_pagination()
    if skip := getattr(namespace, "skip", None):
        pagination.skip = skip
    if limit := getattr(namespace, "limit", None):
        pagination.limit = limit

    query_params["show_deleted"] = show_deleted
    if pipeline_name:
        query_params["pipeline_name"] = pipeline_name
    paginated_raw_pointers: Paginated[dict] = http.get(
        "/v4/pointers", params=dict(**query_params, **pagination.dict())
    ).json()

    pointers = [
        [
            pointer_raw["pointer"],
            pointer_raw["pipeline_id"],
        ]
        for pointer_raw in paginated_raw_pointers["data"]
    ]
    page_position = to_page_position(
        paginated_raw_pointers["skip"],
        paginated_raw_pointers["limit"],
        paginated_raw_pointers["total"],
    )

    table = tabulate(
        pointers,
        headers=["Pointer", "Pipeline ID"],
        tablefmt="psql",
    )
    print(table)
    print(f"\nPage {page_position['current']} of {page_position['total']}\n")


def _create_pointer(namespace: Namespace) -> None:
    new_pointer = getattr(namespace, "new_pointer")
    pipeline_id_or_pointer = getattr(namespace, "pipeline_id_or_pointer")
    locked = getattr(namespace, "locked", False)

    create_schema = pointers_schema.PointerCreate(
        pointer=new_pointer,
        pointer_or_pipeline_id=pipeline_id_or_pointer,
        locked=locked,
    )
    result = http.post(
        "/v4/pointers",
        json.loads(
            create_schema.json(),
        ),
    )

    pointer = pointers_schema.PointerGet.parse_obj(result.json())

    _print(f"Created pointer {pointer.pointer} -> {pointer.pipeline_id}")


def _edit_pointer(namespace: Namespace) -> None:
    pointer = getattr(namespace, "pointer")
    source = getattr(namespace, "source", None)
    locked = getattr(namespace, "locked", None)
    unlocked = getattr(namespace, "unlocked", None)

    if locked and unlocked:
        _print("Cannot lock and unlock at the same time!", level="ERROR")
        return

    if source is None and not locked and not unlocked:
        _print("Nothing to edit!", level="ERROR")
        return

    _print(f"Editing pointer ({pointer})")

    edit_schema = pointers_schema.PointerPatch(
        pointer_or_pipeline_id=source,
        locked=locked if locked is not None else not unlocked,
    )
    http.patch(
        f"/v4/pointers/{pointer}",
        json.loads(
            edit_schema.json(),
        ),
    )

    _print("Pointer edited!")


def _delete_pointer(namespace: Namespace) -> None:
    pointer = getattr(namespace, "pointer")

    _print(f"Deleting pointer ({pointer})")

    http.delete(
        f"/v4/pointers/{pointer}",
    )

    _print("Pointer deleted!")
