import io
import typing as t

import httpx
import requests
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from tqdm import tqdm

from pipeline import current_configuration
from pipeline.util.logging import PIPELINE_STR

ACTIVE_IP = (
    active_remote.url
    if (active_remote := current_configuration.active_remote) is not None
    else None
)

_client = httpx.Client(
    base_url=ACTIVE_IP,
    headers={
        "Authorization": f"Bearer {current_configuration.active_remote.token}",
    },
    timeout=300,
)


def post(
    endpoint: str,
    json_data: dict = None,
) -> httpx.Response:
    try:
        response = _client.post(
            endpoint,
            json=json_data,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise Exception(f"HTTP error: {exc.response.status_code}, {exc.response.text}")

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
            },
        )
    else:
        response = _client.post(
            endpoint,
            files=files,
            params=params,
        )

    return response
