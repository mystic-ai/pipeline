import json
from argparse import ArgumentParser, _SubParsersAction

from tabulate import tabulate

from pipeline import PipelineCloud

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
    remote_service = PipelineCloud(verbose=False)
    resource_information = remote_service._get("/v3/core/resources")
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
            "N/A"
            if resource["gpus"] is None
            else [gpu["name"].strip() for gpu in resource["gpus"]],
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
            "GPUs",
        ],
        tablefmt="psql",
    )

    print(table)
