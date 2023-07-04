import io
import os
import typing as t

import httpx

# from httpx import _GeneratorContextManager
import requests
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from tqdm import tqdm

from pipeline import current_configuration
from pipeline.util.logging import PIPELINE_STR

ACTIVE_IP = (
    active_remote.url
    if (active_remote := current_configuration.active_remote) is not None
    else os.environ.get("PIPELINE_API_URL", None)
)

ACTIVE_TOKEN = (
    current_configuration.active_remote.token
    if current_configuration.active_remote is not None
    else os.environ.get("PIPELINE_API_TOKEN", None)
)
_client = httpx.Client(
    base_url=ACTIVE_IP,
    headers={
        "Authorization": f"Bearer {ACTIVE_TOKEN}",
    },
    timeout=300,
)


def post(
    endpoint: str,
    json_data: dict = None,
    raise_for_status: bool = True,
) -> httpx.Response:
    response = _client.post(
        endpoint,
        json=json_data,
    )
    if raise_for_status:
        response.raise_for_status()

    return response


def get(
    endpoint: str,
) -> httpx.Response:
    response = _client.get(
        endpoint,
    )
    response.raise_for_status()

    return response


def create_callback(encoder: MultipartEncoder) -> t.Callable:
    encoder_len = encoder.len
    bar = tqdm(
        desc=f"{PIPELINE_STR} Uploading",
        unit="B",
        unit_scale=True,
        total=encoder_len,
        unit_divisor=1024,
    )

    def callback(monitor):
        bar.n = monitor.bytes_read
        bar.refresh()
        if monitor.bytes_read == encoder_len:
            bar.close()

    return callback


def get_progress_bar_uploader(
    files: t.Dict[str, io.BufferedIOBase], params: t.Dict[str, str]
) -> MultipartEncoderMonitor:
    encoder = create_upload(files, params)
    callback = create_callback(encoder)
    monitor = MultipartEncoderMonitor(encoder, callback)
    return monitor


def create_upload(
    files: t.Dict[str, io.BufferedIOBase], params: t.Dict[str, str]
) -> MultipartEncoder:
    file_dict = {
        param_key: (
            getattr(file, "name", "undefined"),
            file,
            "application/octet-stream",
            {
                "Content-Transfer-Encoding": "binary",
            },
        )
        for param_key, file in files.items()
    }

    return MultipartEncoder(
        dict(
            **file_dict,
            **params,
        ),
    )


def post_files(
    endpoint: str,
    files: t.Dict[str, t.Any],
    params: t.Dict[str, t.Any] = dict(),
    progress: bool = False,
) -> httpx.Response:
    if progress:
        monitor = get_progress_bar_uploader(files=files, params=params)
        response = requests.post(
            f"{ACTIVE_IP}{endpoint}",
            data=monitor,
            headers={
                "content-Type": monitor.content_type,
                "Authorization": f"Bearer {current_configuration.active_remote.token}",
            },
        )
    else:
        response = _client.post(
            endpoint,
            files=files,
            params=params,
        )

    return response


def stream_post(
    endpoint: str,
    json_data: dict = None,
) -> t.Iterator[httpx.Response]:
    return _client.stream(
        "POST",
        endpoint,
        json=json_data,
    )
