import json
import urllib.parse
from argparse import ArgumentParser, Namespace, _SubParsersAction

from websockets.exceptions import ConnectionClosedError
from websockets.sync.client import connect

from pipeline import current_configuration
from pipeline.util.logging import _print, _print_remote_log


def run_logs_parser(command_parser: "_SubParsersAction[ArgumentParser]") -> None:
    run_logs_parser = command_parser.add_parser("run", help="Get logs for a run.")
    run_logs_parser.set_defaults(func=_run_logs)

    run_logs_parser.add_argument(
        "-f",
        "--follow",
        action="store_true",
    )

    run_logs_parser.add_argument(
        "run_id",
        help="Run ID to get logs for.",
        type=str,
    )


def _run_logs(args: Namespace) -> None:
    follow = getattr(args, "follow", False)
    run_id = getattr(args, "run_id")

    url_info = urllib.parse.urlparse(current_configuration.active_remote.url)
    target_host = url_info.hostname
    target_port = url_info.port

    connection_string = urllib.parse.urljoin(
        "ws://" + target_host + ":" + str(target_port),
        f"/v3/logs/run/{run_id}?follow={follow}",
    )

    try:
        with connect(
            connection_string,
            additional_headers={
                "Authorization": f"Bearer {current_configuration.active_remote.token}"
            },
        ) as websocket:
            message = websocket.recv()
            _print(f"Showing logs for run {run_id}")
            while True:
                stream_data: dict = json.loads(json.loads(message))
                streams = stream_data.get("streams", None)
                if streams is None:
                    raise Exception("No streams in message")

                if len(streams) == 0:
                    raise Exception("No streams in message")

                for stream in streams:
                    values = stream.get("values", None)
                    if values is None:
                        raise Exception("No values in stream")

                    for value in values:
                        _print_remote_log(
                            value,
                        )
                if not follow:
                    break
                message = websocket.recv()
    except ConnectionClosedError as e:
        if e.code == 4000:
            _print(f"Run {run_id} not found", level="ERROR")
