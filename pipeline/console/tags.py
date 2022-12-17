import argparse
import re

from pipeline import PipelineCloud
from pipeline.schemas.pipeline import (
    PipelineTagCreate,
    PipelineTagGet,
    PipelineTagPatch,
)
from pipeline.util.logging import _print


def tags(args: argparse.Namespace) -> int:
    sub_command = getattr(args, "sub-command", None)

    remote_service = PipelineCloud(verbose=False)
    remote_service.authenticate()

    tag_re_pattern = re.compile(r"^[a-zA-Z\-\_]+:[a-zA-Z\-\_]+$", re.IGNORECASE)

    if sub_command in ["create", "update"]:
        source = getattr(args, "source")
        target = getattr(args, "target")

        if not tag_re_pattern.match(source):
            _print("Source tag must match pattern 'pipeline:tag'", level="ERROR")
            return 1

        if tag_re_pattern.match(target):
            # Pointing to another tag
            target_schema = PipelineTagGet.parse_obj(
                remote_service._get(f"/v2/pipeline-tags/by-name/{target}")
            )
            target_pipeline = target_schema.pipeline_id
        else:
            # Pointing to a pipeline_id
            target_pipeline = target

        if sub_command == "create":
            tag_create_schema = PipelineTagCreate(
                name=source,
                pipeline_id=target_pipeline,
            )

            response = remote_service._post(
                "/v2/pipeline-tags", json_data=tag_create_schema.dict()
            )
        else:
            # Update
            existing_tag = PipelineTagGet.parse_obj(
                remote_service._get(f"/v2/pipeline-tags/by-name/{source}")
            )
            patch_schema = PipelineTagPatch(pipeline_id=target_pipeline)

            response = remote_service._patch(
                f"/v2/pipeline-tags/{existing_tag.id}", json_data=patch_schema.dict()
            )

        tag_get_schema = PipelineTagGet.parse_obj(response)

        target_string = f"{target}/" if target_pipeline is not target else None
        _print(f"Tag '{source}' -> '{target_string}{tag_get_schema.pipeline_id}'")

        return 0

    elif sub_command in ["list", "ls"]:
        pipeline_id: str = getattr(args, "pipeline_id", None)

        if pipeline_id is not None:
            ...
        else:
            ...
        return 1
    elif sub_command in ["delete", "rm"]:
        return 1
    elif sub_command == "get":
        return 1
