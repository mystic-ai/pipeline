import requests

from npu2 import api
from npu2.api.exceptions import RouteNotFound

def __handle_response__(response: requests.Response):
    if response.status_code == 404:
        raise RouteNotFound(response)


def post(endpoint, json_data):
    headers = {
        "Authorization": "Bearer %s" % api.API_TOKEN
    }
    
    response = requests.post(api.API_ENDPOINT + endpoint, headers=headers, json=json_data)
    __handle_response__(response)
    return response.json()

def get(endpoint):
    headers = {
        "Authorization": "Bearer %s" % api.API_TOKEN
    }
    
    response = requests.get(api.API_ENDPOINT + endpoint, headers=headers)
    __handle_response__(response)
    return response.json()