import urllib.parse
import requests

from pipeline.logging import _print


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
