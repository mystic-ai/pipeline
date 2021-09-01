import requests

from npu2 import api, npu_print
from npu2.api import API_ENDPOINT

def link(api_token: str):
    api.API_TOKEN = api_token
    
    response= requests.get(API_ENDPOINT + "/status")
    if response.json()["alive"]:
        npu_print ("Succesfully authenticated with the Neuro API")
        