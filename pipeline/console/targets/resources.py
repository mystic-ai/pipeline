import json
from argparse import ArgumentParser, _SubParsersAction

from tabulate import tabulate

from pipeline.cloud import http
from pipeline.cloud.compute_requirements import Accelerator

# Parser builder


def create_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    ...


def get_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    get_parser = command_parser.add_parser(
        "resources", help="Get resource information."
    )
    get_parser.set_defaults(func=lambda _: list_resources())


def delete_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    ...


# Functions


def _shorten_id(id: str, leading_length: int = 4, trailing_length: int = 4) -> str:
    if id == "None":
        return "None"
    return id[:leading_length] + "..." + id[-trailing_length:]


def list_resources() -> None:
    resource_information = http.get("/v3/core/resources").json()
    resource_information = [json.loads(resource) for resource in resource_information]

    resource_data = [
        [
            _shorten_id(resource["id"]),
            [
                _shorten_id(p_id)
                for cached_pipelines in resource["pipeline_cache"].values()
                for p_id in cached_pipelines
            ],
            _shorten_id(str(resource["current_run"]))
            if (resource["busy"] == 1 and resource["current_run"] != -1)
            else "-",
            [_shorten_id(id) for id in resource["run_queue"]],
            "cpu"
            if not (accelerators := resource.get("gpus", None))
            else (
                "cpu"
                if "cpu" in accelerators
                else "\n".join(
                    [
                        f"{[accel['name'] for accel in accelerators].count(accelerator)}× {Accelerator.from_str(accelerator)} ({round(sum([accel['vram_total_mb'] for accel in accelerators if accel['name'] == accelerator]) / 1024.0, 1)}GB VRAM)"  # noqa E501
                        for accelerator in set(
                            [accel["name"] for accel in accelerators]
                        )
                    ]
                )
            )
            # "N/A"
            # if resource["gpus"] is None
            # else [gpu["name"].strip() for gpu in resource["gpus"]],
        ]
        for resource in resource_information
    ]

    table = tabulate(
        resource_data,
        headers=[
            "ID",
            "Pipelines",
            "Current run",
            "Run queue",
            "Accelerators",
        ],
        tablefmt="psql",
    )

    print(table)
