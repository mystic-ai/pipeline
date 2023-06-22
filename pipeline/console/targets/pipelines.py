import typing as t
from argparse import ArgumentParser, Namespace, _SubParsersAction
from datetime import datetime

from tabulate import tabulate

from pipeline.api import PipelineCloud
from pipeline.util.logging import _print
from pipeline.v3.schemas import pipelines as pipelines_schema


def edit_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    edit_parser = command_parser.add_parser(
        "pipelines",
        aliases=["pipeline"],
        help="Edit a pipeline.",
    )

    edit_parser.set_defaults(func=_edit_pipeline)

    edit_parser.add_argument(
        "pipeline_id",
        help="Pipeline to edit.",
    )

    edit_parser.add_argument(
        "--cache-number",
        "-c",
        help="Minimum cache number.",
        type=int,
    )

    edit_parser.add_argument(
        "--gpu-memory",
        "-g",
        help="Minimum GPU memory.",
        type=int,
    )


def get_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    get_parser = command_parser.add_parser(
        "pipelines",
        aliases=["pipeline"],
        help="Get pipeline information.",
    )

    get_parser.set_defaults(func=_get_pipeline)


def delete_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    ...


def _get_pipeline(args: Namespace) -> None:
    _print("Getting pipelines")

    cluster_api = PipelineCloud(verbose=False)
    pipelines_raw: t.List[dict] = cluster_api._get("/v3/pipelines")

    pipelines = [
        [
            pipeline_raw["id"],
            datetime.fromtimestamp(pipeline_raw.get("created_at"))
            if "created_at" in pipeline_raw
            else "N/A",
            datetime.fromtimestamp(pipeline_raw.get("updated_at"))
            if "updated_at" in pipeline_raw
            else "N/A",
            val if (val := pipeline_raw.get("minimum_cache_number", "N/A")) else "N/A",
            val if (val := pipeline_raw.get("gpu_memory_min", "N/A")) else "N/A",
        ]
        for pipeline_raw in pipelines_raw
    ]

    table = tabulate(
        pipelines,
        headers=[
            "ID",
            "Created At",
            "Updated At",
            "Cache number",
            "GPU memory",
        ],
        tablefmt="outline",
    )
    print(table)


def _edit_pipeline(args: Namespace) -> None:
    pipeline_id = getattr(args, "pipeline_id")
    cache_number = getattr(args, "cache_number", None)
    gpu_memory = getattr(args, "gpu_memory", None)

    patch_schema = pipelines_schema.PipelinePatch(
        minimum_cache_number=cache_number,
        gpu_memory_min=gpu_memory,
    )

    if cache_number is None and gpu_memory is None:
        _print("Nothing to edit.", level="ERROR")
        return

    cluster_api = PipelineCloud(verbose=False)
    cluster_api._patch(
        f"/v3/pipelines/{pipeline_id}",
        patch_schema.dict(),
    )

    _print("Pipeline edited!")