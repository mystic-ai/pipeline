import argparse
import json

from tabulate import tabulate

from pipeline.api import PipelineCloud
from pipeline.schemas.file import FileGet
from pipeline.schemas.pagination import Paginated
from pipeline.schemas.run import RunGet, RunState
from pipeline.util import hex_to_python_object


def runs(args: argparse.Namespace) -> int:
    sub_command = getattr(args, "sub-command", None)

    remote_service = PipelineCloud(verbose=False)

    if sub_command in ["list", "ls"]:
        raw_result = remote_service.get_runs()

        schema = Paginated[RunGet].parse_obj(raw_result)

        runs = schema.data

        terminal_run_states = [
            RunState.FAILED,
            RunState.COMPLETE,
        ]

        run_data = [
            [
                _run.id,
                _run.created_at.strftime("%d-%m-%Y %H:%M:%S"),
                "executing",
                _run.runnable.id,
            ]
            for _run in runs
            if _run.run_state not in terminal_run_states
        ]
        table = tabulate(
            run_data,
            headers=[
                "ID",
                "Created at",
                "State",
                "Pipeline",
            ],
            tablefmt="outline",
        )
        print(table)
        return 0
    elif sub_command == "get":
        run_id = args.run_id

        result = remote_service._get(f"/v2/runs/{run_id}")
        if args.result:
            result = RunGet.parse_obj(result)
            if result.result_preview is not None:
                print(json.dumps(result.result_preview))
            else:
                file_schema_raw = remote_service._get(
                    f"/v2/files/{result.result.id}?return_data=true"
                )

                file_schema = FileGet.parse_obj(file_schema_raw)
                raw_result = hex_to_python_object(file_schema.data)
                print(json.dumps(raw_result))

            return 0
        else:
            print(json.dumps(result))
            return 0
