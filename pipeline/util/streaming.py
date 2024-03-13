import json
import typing as t

from httpx import Response


def handle_stream_response(response: Response) -> t.Generator[t.Any, t.Any, None]:
    """Helper function to help with parsing streamed data from an API into
    JSON objects
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
                result_json = json.loads(result)
                incomplete_chunk = ""
            except json.JSONDecodeError:
                # assume due to an incomplete chunk of JSON
                incomplete_chunk = result
                break

            yield result_json


async def handle_async_stream_response(
    response: Response,
) -> t.AsyncGenerator[t.Any, t.Any]:
    """Helper function to help with parsing streamed data from an API into
    JSON objects

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
                result_json = json.loads(result)
                incomplete_chunk = ""
            except json.JSONDecodeError:
                # assume due to an incomplete chunk of JSON
                incomplete_chunk = result
                break

            yield result_json
