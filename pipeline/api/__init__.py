import inspect
import requests
import urllib.parse
from dill import dumps

from pipeline.logging import _print
from pipeline.schemas import PipelineFunctionSchema

from pipeline.api.function import upload_function


PIPELINE_API_TOKEN: str = None
PIPELINE_API_URL: str = "https://api.pipeline.ai"


def authenticate(token: str, url: str = PIPELINE_API_URL):
    """
    Authenticate with the pipeline.ai API
    """
    _print("Authenticating")

    global PIPELINE_API_TOKEN
    PIPELINE_API_TOKEN = token

    # TODO: Change this url to an actual auth one, not status which just shows if the API is alive.
    status_url = urllib.parse.urljoin(url, "/status")

    response = requests.get(status_url)
    if response.json()["alive"]:
        _print("Succesfully authenticated with the Neuro API")


def upload(object):

    if hasattr(object, "__pipeline_function__") and isinstance(
        object.__pipeline__function__, PipelineFunctionSchema
    ):
        function_schema: PipelineFunctionSchema = object.__pipeline_function__
        function_name = function_schema.name
        function_bytes = dumps(function_schema.function)
        function_hex = function_bytes.hex()
        function_source = inspect.getsource(function_schema.function)
        return upload_function(function_name, function_hex, function_source)
    else:
        raise Exception("Not a pipeline object!")
