import requests
import urllib.parse

from pipeline import api


def __handle_response__(response: requests.Response):
    if response.status_code == 404:
        raise Exception(response)


def post(endpoint, json_data):
    headers = {
        "Authorization": "Bearer %s" % api.PIPELINE_API_TOKEN,
        "Content-type": "application/json",
    }

    url = urllib.parse.urljoin(api.PIPELINE_API_URL, endpoint)
    response = requests.post(url, headers=headers, json=json_data)
    __handle_response__(response)
    return response.json()


def get(endpoint):
    headers = {"Authorization": "Bearer %s" % api.PIPELINE_API_TOKEN}

    url = urllib.parse.urljoin(api.PIPELINE_API_URL, endpoint)

    response = requests.get(url, headers=headers)
    __handle_response__(response)
    return response.json()
