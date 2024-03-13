import typing as t

from httpx import Response
from pydantic import BaseModel, ValidationError

T = t.TypeVar("T", bound=BaseModel)


def handle_stream_response(
    response: Response, result_schema_cls: T
) -> t.Generator[T, t.Any, None]:
    """Helper function to help with parsing streamed data from an API into
    instances of 'result_schema_cls'
    """
    incomplete_chunk = ""
    for chunk in response.iter_text():
        # chunk of data coming back from API could either be an incomplete
        # result or more than one results, so try to handle both these cases
        full_chunk = incomplete_chunk + chunk
        results = full_chunk.split("\n")
        for result in results:
            if not result:
                continue
            try:
                result = result_schema_cls.parse_raw(result)
                incomplete_chunk = ""
            except ValidationError:
                # assume due to an incomplete chunk of JSON
                incomplete_chunk = result
                break

            yield result


async def handle_async_stream_response(
    response: Response, result_schema_cls: T
) -> t.AsyncGenerator[T, t.Any]:
    """Helper function to help with parsing streamed data from an API into
    instances of 'result_schema_cls'

    Same as above but for async httpx client.
    """
    incomplete_chunk = ""
    async for chunk in response.aiter_text():
        # chunk of data coming back from API could either be an incomplete
        # result or more than one results, so try to handle both these cases
        full_chunk = incomplete_chunk + chunk
        results = full_chunk.split("\n")
        for result in results:
            if not result:
                continue
            try:
                result = result_schema_cls.parse_raw(result)
                incomplete_chunk = ""
            except ValidationError:
                # assume due to an incomplete chunk of JSON
                incomplete_chunk = result
                break

            yield result
