from argparse import ArgumentParser, Namespace, _SubParsersAction
from datetime import datetime

from tabulate import tabulate

from pipeline.cloud import http
from pipeline.cloud.compute_requirements import Accelerator
from pipeline.cloud.schemas import pipelines as pipelines_schema
from pipeline.cloud.schemas.pagination import (
    Paginated,
    get_default_pagination,
    to_page_position,
)
from pipeline.util.logging import _print


def edit_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    edit_parser = command_parser.add_parser(
        "pipelines",
        aliases=["pipeline", "pl"],
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
    edit_parser.add_argument(
        "--scaling-config",
        "-s",
        help="The scaling configuration name the pipeline uses",
        type=str,
        required=False,
    )


def get_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    get_parser = command_parser.add_parser(
        "pipelines",
        aliases=["pipeline", "pl"],
        help="Get pipeline information.",
    )

    get_parser.set_defaults(func=_get_pipeline)

    # get by name
    # get_parser.add_argument(
    #     "--name",
    #     "-n",
    #     help="Pipeline name.",
    #     type=str,
    # )
    get_parser.add_argument(
        "--skip",
        "-s",
        help="Number of pipelines to skip in paginated set.",
        type=int,
    )
    get_parser.add_argument(
        "--limit",
        "-l",
        help="Total number of pipelines to fetch in paginated set.",
        type=int,
    )


def delete_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    delete_parser = command_parser.add_parser(
        "pipelines",
        aliases=["pipeline", "pl"],
        help="Delete a pipeline.",
    )

    delete_parser.set_defaults(func=_delete_pipeline)

    delete_parser.add_argument(
        "pipeline_id",
        help="Pipeline to delete.",
    )


def _get_pipeline(args: Namespace) -> None:
    _print("Getting pipelines")

    # cluster_api = PipelineCloud(verbose=False)

    params = dict()
    pagination = get_default_pagination()
    # if name := getattr(args, "name", None):
    #     params["name"] = name
    if skip := getattr(args, "skip", None):
        pagination.skip = skip
    if limit := getattr(args, "limit", None):
        pagination.limit = limit
    paginated_raw_pipelines: Paginated[dict] = http.get(
        "/v4/pipelines",
        params=dict(**params, **pagination.dict()),
    ).json()

    # pipelines_: Paginated[pipelines_schema.PipelineGet] = Paginated[
    #     pipelines_schema.PipelineGet
    # ].parse_obj(paginated_raw_pipelines)

    pipelines = [
        [
            pipeline_raw["id"],
            pipeline_raw["name"],
            (
                datetime.fromtimestamp(pipeline_raw.get("created_at"))
                if "created_at" in pipeline_raw
                else "N/A"
            ),
            val if (val := pipeline_raw.get("minimum_cache_number", "N/A")) else "N/A",
            (
                ""
                if not (accelerators := pipeline_raw.get("accelerators", None))
                else (
                    "cpu"
                    if Accelerator.cpu in pipeline_raw.get("accelerators", [])
                    else "\n".join(
                        [
                            f"{accelerators.count(accelerator)}× {accelerator}"
                            for accelerator in set(accelerators)
                        ]
                    )
                )
            ),
        ]
        for pipeline_raw in paginated_raw_pipelines["data"]
    ]

    page_position = to_page_position(
        paginated_raw_pipelines["skip"],
        paginated_raw_pipelines["limit"],
        paginated_raw_pipelines["total"],
    )

    table = tabulate(
        pipelines,
        headers=[
            "ID",
            "Name",
            "Created",
            "Cache # (min-max)",
            "Accelerators",
        ],
        tablefmt="psql",
    )
    print(table)
    print(f"\nPage {page_position['current']} of {page_position['total']}\n")


def _edit_pipeline(args: Namespace) -> None:
    pipeline_id = getattr(args, "pipeline_id")
    cache_number = getattr(args, "cache_number", None)
    gpu_memory = getattr(args, "gpu_memory", None)
    scaling_config = getattr(args, "scaling_config", None)

    patch_schema = pipelines_schema.PipelinePatch(
        minimum_cache_number=cache_number,
        gpu_memory_min=gpu_memory,
        scaling_config=scaling_config,
    )
    if all(arg is None for arg in (gpu_memory, cache_number, scaling_config)):
        _print("Nothing to edit.", level="ERROR")
        return

    http.patch(
        f"/v4/pipelines/{pipeline_id}",
        patch_schema.dict(),
    )

    _print("Pipeline edited!")


def _delete_pipeline(args: Namespace) -> None:
    pipeline_id = getattr(args, "pipeline_id")

    http.delete(
        f"/v4/pipelines/{pipeline_id}",
    )

    _print("Pipeline deleted!")
