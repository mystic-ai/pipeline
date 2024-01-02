import functools
import io
import os
import typing as t

import httpx

# from httpx import _GeneratorContextManager
import requests
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from tqdm import tqdm

from pipeline import current_configuration
from pipeline.util.logging import PIPELINE_STR, _print

_client = None
_client_async = None


def get_response_error_dict(e: httpx.HTTPStatusError) -> t.Dict:
    try:
        detail = e.response.json()["detail"]
        if not isinstance(detail, dict):
            detail = {"detail": detail}
    except (TypeError, KeyError):
        return {"message": "Something went wrong.", "response_json": e.response.json()}
    return {**detail, "request_id": e.response.headers.get("x-correlation-id")}


def handle_http_status_error(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        response = func(*args, **kwargs)
        if kwargs.pop("handle_error", True):
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                _print(
                    get_response_error_dict(e),
                    level="ERROR",
                )
                raise
        return response

    return wrapper


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        url = (
            active_remote.url
            if (active_remote := current_configuration.active_remote) is not None
            else os.environ.get("PIPELINE_API_URL", "https://www.mystic.ai/")
        )
        token = (
            current_configuration.active_remote.token
            if current_configuration.active_remote is not None
            else os.environ.get("PIPELINE_API_TOKEN", None)
        )
        _client = httpx.Client(
            base_url=url,
            headers={
                "Authorization": f"Bearer {token}",
            },
            timeout=300,
        )
    return _client


def _get_async_client() -> httpx.AsyncClient:
    global _client_async
    if _client_async is None:
        url = (
            active_remote.url
            if (active_remote := current_configuration.active_remote) is not None
            else os.environ.get("PIPELINE_API_URL", "https://www.mystic.ai/")
        )
        token = (
            current_configuration.active_remote.token
            if current_configuration.active_remote is not None
            else os.environ.get("PIPELINE_API_TOKEN", None)
        )
        _client_async = httpx.AsyncClient(
            base_url=url,
            headers={
                "Authorization": f"Bearer {token}",
            },
            timeout=300,
        )
    return _client_async


@handle_http_status_error
def post(
    endpoint: str,
    json_data: dict = None,
    handle_error: bool = True,
) -> httpx.Response:
    client = _get_client()
    response = client.post(endpoint, json=json_data)
    return response


@handle_http_status_error
async def async_post(
    endpoint: str,
    json_data: dict = None,
    handle_error: bool = True,
) -> httpx.Response:
    client = _get_async_client()
    response = await client.post(
        endpoint,
        json=json_data,
    )
    return response


@handle_http_status_error
def patch(
    endpoint: str,
    json_data: dict = None,
    handle_error: bool = True,
) -> httpx.Response:
    client = _get_client()
    return client.patch(
        endpoint,
        json=json_data,
    )


@handle_http_status_error
def get(
    endpoint: str,
    handle_error: bool = True,
    **kwargs,
) -> httpx.Response:
    client = _get_client()
    response = client.get(endpoint, **kwargs)
    return response


@handle_http_status_error
def delete(
    endpoint: str,
    handle_error: bool = True,
    **kwargs,
) -> httpx.Response:
    client = _get_client()
    response = client.delete(endpoint, **kwargs)
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
    data: t.Dict[str, t.Any] = dict(),
    progress: bool = False,
) -> httpx.Response:
    if progress:
        monitor = get_progress_bar_uploader(files=files, params=params)
        url = (
            active_remote.url
            if (active_remote := current_configuration.active_remote) is not None
            else os.environ.get("PIPELINE_API_URL", "https://www.mystic.ai/")
        )
        token = (
            current_configuration.active_remote.token
            if current_configuration.active_remote is not None
            else os.environ.get("PIPELINE_API_TOKEN", None)
        )
        response = requests.post(
            f"{url}{endpoint}",
            data=monitor,
            headers={
                "content-Type": monitor.content_type,
                "Authorization": f"Bearer {token}",
            },
            params=params,
        )
    else:
        client = _get_client()
        response = client.post(
            endpoint,
            files=files,
            data=data,
            params=params,
        )

    return response
