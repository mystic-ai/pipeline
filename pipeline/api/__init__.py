import os
import inspect
import requests
import urllib.parse
from dill import dumps

from pipeline.logging import _print
from pipeline.schemas import PipelineGraph
from pipeline.objects import PipelineFunction

from pipeline.api.function import upload_function
from pipeline.api.pipeline import upload_pipeline
from pipeline.api.model import upload_model

from pipeline.pipeline_schemas.function import FunctionGet

PIPELINE_API_TOKEN: str = None
PIPELINE_API_URL: str = os.getenv("PIPELINE_API_URL", "https://api.pipeline.ai")


def authenticate(token: str, url: str = PIPELINE_API_URL):
    """
    Authenticate with the pipeline.ai API
    """
    _print("Authenticating")

    global PIPELINE_API_TOKEN
    PIPELINE_API_TOKEN = token
    global PIPELINE_API_URL
    PIPELINE_API_URL = url
    # TODO: Change this url to an actual auth one, not status which just shows if the API is alive.
    status_url = urllib.parse.urljoin(url, "/v2/status")

    response = requests.get(status_url)
    if response.json()["alive"]:
        _print(
            "Succesfully authenticated with the Pipeline API (%s)" % PIPELINE_API_URL
        )


def upload(object):

    if (
        hasattr(object, "__function__")
        and hasattr(object.__function__, "__pipeline_function__")
        and isinstance(object.__function__.__pipeline_function__, PipelineFunction)
    ):
        function: PipelineFunction = object.__function__.__pipeline_function__

        return upload_function(function._api_create_schema)
    elif isinstance(object, PipelineGraph):
        pipeline_graph = object

        for fi, function in enumerate(pipeline_graph.functions):
            upload_result = FunctionGet.parse_obj(
                upload_function(function._api_create_schema)
            )
            pipeline_graph.functions[fi]._api_get_schema = upload_result

        for model in pipeline_graph.models:
            upload_model(model)

        upload_pipeline(pipeline_graph._api_create_schema)
        return pipeline_graph
    else:
        raise Exception("Not a pipeline object!")
