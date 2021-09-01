import requests

from npu2 import api

def post(endpoint, json_data):
    headers = {
        "Authorization": "Bearer %s" % api.API_TOKEN
    }
    
    response = requests.post(api.API_ENDPOINT + endpoint, headers=headers, json=json_data)
    return response.json()

def get(endpoint):
    headers = {
        "Authorization": "Bearer %s" % api.API_TOKEN
    }
    
    response = requests.get(api.API_ENDPOINT + endpoint, headers=headers)
    return response.json()