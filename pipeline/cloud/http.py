import functools
import io
import os
import typing as t
from contextlib import contextmanager
from json.decoder import JSONDecodeError

import httpx
import requests
from fastapi.responses import StreamingResponse
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from starlette.types import Send
from tqdm import tqdm

from pipeline import current_configuration
from pipeline.util.logging import PIPELINE_STR, _print

_client = None
_client_async = None


class APIError(Exception):
    def __init__(
        self, url: str, status_code: int, detail: dict | str, request_id: str | None
    ):
        self.url = url
        self.status_code = status_code
        self.detail = detail
        self.request_id = request_id
        super().__init__(url, status_code, detail, request_id)

    @classmethod
    def from_response(cls, response: httpx.Response):
        try:
            detail = response.json()["detail"]
            if not isinstance(detail, dict):
                detail = {"detail": detail}
        except (JSONDecodeError, TypeError, KeyError):
            detail = {
                "message": "Something went wrong.",
                "response": response.text,
            }
        return cls(
            url=str(response.url),
            detail=detail,
            status_code=response.status_code,
            request_id=response.headers.get("x-correlation-id"),
        )

    def __str__(self):
        error_msg = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        error_msg = f"APIError({error_msg})"
        return error_msg


def raise_if_http_status_error(response: httpx.Response):
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        error = APIError.from_response(e.response)
        _print(str(error), level="ERROR")
        raise error from None


def handle_http_status_error(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        response = func(*args, **kwargs)
        if kwargs.pop("handle_error", True):
            raise_if_http_status_error(response)
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
def post_file(endpoint: str, files):
    client = _get_client()
    response = client.post(endpoint, files=files)
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


@handle_http_status_error
@contextmanager
def stream(
    method: str,
    endpoint: str,
    json_data: dict | None = None,
    handle_error: bool = True,
):
    client = _get_client()
    with client.stream(method=method, url=endpoint, json=json_data) as response:
        yield response


class StreamingResponseWithStatusCode(StreamingResponse):
    """
    Variation of StreamingResponse that can dynamically decide the HTTP status
    code, based on the returns from the content iterator (parameter 'content').
    Expects the content to yield tuples of (content: str, status_code: int),
    instead of just content as it was in the original StreamingResponse. The
    parameter status_code in the constructor is ignored, but kept for
    compatibility with StreamingResponse.

    See
    https://github.com/tiangolo/fastapi/discussions/10138#discussioncomment-8216436
    for inspiration
    """

    async def stream_response(self, send: Send) -> None:
        first_chunk_content, self.status_code = await anext(self.body_iterator)
        if not isinstance(first_chunk_content, bytes):
            first_chunk_content = first_chunk_content.encode(self.charset)

        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": first_chunk_content,
                "more_body": True,
            }
        )

        # ignore status code after response has started
        async for chunk_content, _ in self.body_iterator:
            if not isinstance(chunk_content, bytes):
                chunk = chunk_content.encode(self.charset)
            await send({"type": "http.response.body", "body": chunk, "more_body": True})

        await send({"type": "http.response.body", "body": b"", "more_body": False})
