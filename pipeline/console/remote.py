import argparse
import json

from tabulate import tabulate

from pipeline import PipelineCloud, current_configuration
from pipeline.exceptions.MissingActiveToken import MissingActiveToken
from pipeline.util.logging import _print


def remote(args: argparse.Namespace) -> int:
    sub_command = getattr(args, "sub-command", None)

    if sub_command == "set":
        alias = args.alias

        current_configuration.set_active_remote(alias)
        _print(f"Set new default remote to '{alias}'")
        return 0
    elif sub_command in ["list", "ls"]:
        remotes = [
            f"{_remote} (active)" if _remote.active else f"{_remote}"
            for _remote in current_configuration.remotes
        ]
        _print("Authenticated remotes:")
        [print(_remote) for _remote in remotes]
        return 0
    elif sub_command == "login":
        try:
            PipelineCloud(token=args.token, url=args.url, verbose=False)
        except MissingActiveToken:
            _print(f"Couldn't authenticate with {args.url}", level="ERROR")
            return 1

        current_configuration.add_remote(
            alias=args.alias,
            url=args.url,
            token=args.token,
        )

        if args.active:
            _print(f"Setting new remote as active '{args.alias}'")
            current_configuration.set_active_remote(args.alias)

        _print(f"Successfully authenticated with {args.alias}")
        return 0
    elif sub_command == "resources":
        remote_service = PipelineCloud(verbose=False)
        resource_information = remote_service._get("/v3/core/resources")
        resource_information = [
            json.loads(resource) for resource in resource_information
        ]

        resource_data = [
            [
                resource["id"],
                str(resource["current_run"])
                if (resource["busy"] == 1 and resource["current_run"] != -1)
                else "-",
                [
                    p_id
                    for cached_pipelines in resource["pipeline_cache"].values()
                    for p_id in cached_pipelines
                ],
                resource["run_queue"],
                "N/A"
                if resource["gpus"] is None
                else [gpu["name"].strip() for gpu in resource["gpus"]],
            ]
            for resource in resource_information
        ]
        table = tabulate(
            resource_data,
            headers=["ID", "Current run", "Cache", "Queue", "GPUs"],
            tablefmt="outline",
        )

        print(table)
        return 0
