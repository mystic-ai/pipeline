import json

from pipeline.cloud import http
from pipeline.cloud.schemas import pointers as pointers_schemas
from pipeline.util.logging import _print


def create_pointer(
    new_pointer: str,
    pointer_or_pipeline_id: str,
    force=False,
) -> None:
    create_schema = pointers_schemas.PointerCreate(
        pointer=new_pointer,
        pointer_or_pipeline_id=pointer_or_pipeline_id,
        locked=False,
    )
    result = http.post(
        "/v4/pointers",
        json.loads(
            create_schema.json(),
        ),
        handle_error=False,
    )

    if result.status_code == 409:
        if force:
            _print("Pointer already exists, forcing update", "WARNING")
            _edit_pointer(new_pointer, pointer_or_pipeline_id)
        else:
            _print(
                f"Pointer {new_pointer} already exists, use --pointer-overwrite to update",  # noqa
                "WARNING",
            )
        return
    elif result.status_code == 201:
        pointer = pointers_schemas.PointerGet.parse_obj(result.json())
        _print(f"Created pointer {pointer.pointer} -> {pointer.pipeline_id}")
    else:
        raise ValueError(f"Failed to create pointer {new_pointer}\n{result.text}")


def _edit_pointer(
    existing_pointer: str,
    pointer_or_pipeline_id: str,
):
    edit_schema = pointers_schemas.PointerPatch(
        pointer_or_pipeline_id=pointer_or_pipeline_id,
        locked=False,
    )

    result = http.patch(
        f"/v4/pointers/{existing_pointer}",
        json.loads(
            edit_schema.json(),
        ),
    )

    if result.status_code == 200:
        pointer = pointers_schemas.PointerGet.parse_obj(result.json())
        _print(f"Updated pointer {pointer.pointer} -> {pointer.pipeline_id}")
    else:
        _print(f"Failed to edit pointer {existing_pointer}", "ERROR")
