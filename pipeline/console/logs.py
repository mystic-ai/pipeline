from argparse import ArgumentParser, Namespace, _SubParsersAction

from pipeline.cloud.logs import get_run_logs


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
    # TODO: Need to add back in, currently always follows
    # follow = getattr(args, "follow", False)
    run_id = getattr(args, "run_id")

    log_entries = get_run_logs(run_id)
    if not log_entries:
        return
    for message in log_entries:
        print(message)
