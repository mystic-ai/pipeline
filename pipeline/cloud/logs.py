import json
import urllib.parse
from typing import Generator

from websockets.exceptions import ConnectionClosedError
from websockets.sync.client import connect

from pipeline import current_configuration
from pipeline.util.logging import _print


def tail_run_logs(run_id: str) -> Generator[tuple, None, None]:
    url_info = urllib.parse.urlparse(current_configuration.active_remote.url)
    target_host = url_info.hostname
    target_port = url_info.port
    if target_port is None:
        target_port = 80 if url_info.scheme == "http" else 443

    connection_string = urllib.parse.urljoin(
        "ws://" + str(target_host) + ":" + str(target_port),
        f"/v3/logs/run/{run_id}?follow=true",
    )
    try:
        with connect(
            connection_string,
            additional_headers={
                "Authorization": f"Bearer {current_configuration.active_remote.token}"
            },
        ) as websocket:
            message = websocket.recv()
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
                        yield value
                message = websocket.recv()
    except ConnectionClosedError as e:
        if e.code == 4000:
            _print(f"Run {run_id} not found", level="ERROR")