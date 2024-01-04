from argparse import ArgumentParser, Namespace, _SubParsersAction

from pipeline.cloud.logs import get_pipeline_startup_logs, get_run_logs


def run_logs_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    run_logs_parser = command_parser.add_parser("run", help="Get logs for a run.")
    run_logs_parser.set_defaults(func=_run_logs)

    # not supported currently
    # run_logs_parser.add_argument(
    #     "-f",
    #     "--follow",
    #     action="store_true",
    # )

    run_logs_parser.add_argument(
        "run_id",
        help="Run ID to get logs for.",
        type=str,
    )


def _run_logs(args: Namespace) -> None:
    # follow = getattr(args, "follow", False)
    run_id = getattr(args, "run_id")

    log_entries = get_run_logs(run_id)
    if not log_entries:
        return
    for message in log_entries:
        print(message)


def pipeline_startup_logs_parser(
    command_parser: "_SubParsersAction[ArgumentParser]",
) -> None:
    startup_logs_parser = command_parser.add_parser(
        "startup", help="Get logs for a pipeline during startup."
    )
    startup_logs_parser.set_defaults(func=_pipeline_startup_logs)

    # not supported currently
    # startup_logs_parser.add_argument(
    #     "-f",
    #     "--follow",
    #     action="store_true",
    # )

    startup_logs_parser.add_argument(
        "pipeline_id_or_pointer",
        help="Pipeline ID or pointer to get logs for.",
        type=str,
    )


def _pipeline_startup_logs(args: Namespace) -> None:
    # follow = getattr(args, "follow", False)
    pipeline_id_or_pointer = getattr(args, "pipeline_id_or_pointer")

    log_entries = get_pipeline_startup_logs(pipeline_id_or_pointer)
    if not log_entries:
        print("No logs found in the last 24h")
        return
    for message in log_entries:
        print(message)
