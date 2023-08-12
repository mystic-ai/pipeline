import json
import re
from argparse import ArgumentParser, Namespace, _SubParsersAction

from httpx import HTTPStatusError
from tabulate import tabulate

from pipeline.cloud import http
from pipeline.cloud.schemas import pointers as pointers_schema
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
    query_params["show_deleted"] = show_deleted
    if pipeline_name:
        query_params["pipeline_name"] = pipeline_name
    pointers_raw = http.get("/v3/pointers", params=query_params).json()

    pointers = [
        [
            pointer_raw["pointer"],
            pointer_raw["pipeline_id"],
        ]
        for pointer_raw in pointers_raw
    ]

    table = tabulate(
        pointers,
        headers=["Pointer", "Pipeline ID"],
        tablefmt="psql",
    )
    print(table)


def _create_pointer(namespace: Namespace) -> None:
    new_pointer = getattr(namespace, "new_pointer")
    pipeline_id_or_pointer = getattr(namespace, "pipeline_id_or_pointer")
    locked = getattr(namespace, "locked", False)

    create_schema = pointers_schema.PointerCreate(
        pointer=new_pointer,
        pointer_or_pipeline_id=pipeline_id_or_pointer,
        locked=locked,
    )
    try:
        result = http.post(
            "/v3/pointers",
            json.loads(
                create_schema.json(),
            ),
        )

        pointer = pointers_schema.PointerGet.parse_obj(result.json())

        _print(f"Created pointer {pointer.pointer} -> {pointer.pipeline_id}")
    except HTTPStatusError as e:
        if e.response.status_code == 409:
            _print("Pointer already exists!", level="ERROR")
            return
        else:
            raise e


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
        f"/v3/pointers/{pointer}",
        json.loads(
            edit_schema.json(),
        ),
    )

    _print("Pointer edited!")


def _delete_pointer(namespace: Namespace) -> None:
    pointer = getattr(namespace, "pointer")

    _print(f"Deleting pointer ({pointer})")

    http.delete(
        f"/v3/pointers/{pointer}",
    )

    _print("Pointer deleted!")
