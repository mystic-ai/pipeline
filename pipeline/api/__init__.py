import requests

import urllib.parse

import os

import requests
import urllib.parse

from requests.api import head

from pipeline.util.logging import _print

# from pipeline.objects.graph import Graph
# from pipeline.objects.function import Function

# from pipeline.api.function import upload_function
# from pipeline.api.pipeline import upload_pipeline
# from pipeline.schemas.function import FunctionGet


PIPELINE_API_TOKEN: str = None


# PIPELINE_API_TOKEN: str = None
PIPELINE_API_URL: str = os.getenv("PIPELINE_API_URL", "https://api.pipeline.ai")


def __handle_response__(response: requests.Response):
    if response.status_code == 404:

        raise Exception(response.text)
    elif response.status_code == 422:
        raise Exception(response.text)
    elif response.status_code == 500:
        raise Exception(response.text)


def authenticate(token: str, url: str = PIPELINE_API_URL):
    """
    Authenticate with the pipeline.ai API
    """
    _print("Authenticating")
    if token is None:
        raise Exception("Must input a token")
    global PIPELINE_API_TOKEN
    PIPELINE_API_TOKEN = token
    global PIPELINE_API_URL
    PIPELINE_API_URL = url
    # TODO: Change this url to an actual auth one, not status which just shows if the API is alive.
    status_url = urllib.parse.urljoin(url, "/v2/users/me")

    response = requests.get(
        status_url, headers={"Authorization": "Bearer %s" % PIPELINE_API_TOKEN}
    )

    __handle_response__(response)

    if response.json():
        _print(
            "Succesfully authenticated with the Pipeline API (%s)" % PIPELINE_API_URL
        )


"""
def upload(object):

    if (
        hasattr(object, "__function__")
        and hasattr(object.__function__, "__pipeline_function__")
        and isinstance(object.__function__.__pipeline_function__, Function)
    ):
        function: Function = object.__function__.__pipeline_function__

        return upload_function(function.to_create_schema())
    elif isinstance(object, Graph):
        pipeline_graph = object

        for fi, function in enumerate(pipeline_graph.functions):
            upload_result = FunctionGet.parse_obj(
                upload_function(function.to_create_schema())
            )
            # TODO: Update foreign id after upload
            # pipeline_graph.functions[fi]._api_get_schema = upload_result

        # for model in pipeline_graph.models:
        #    upload_model(model)

        upload_pipeline(pipeline_graph.to_create_schema())
        return pipeline_graph
    else:
        raise Exception("Not a pipeline object!")
"""
