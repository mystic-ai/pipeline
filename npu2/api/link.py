from npu2.api import API_ENDPOINT
import requests

from npu2 import api 

def link(api_token: str):
    api.API_TOKEN = api_token
    
    response= requests.get(API_ENDPOINT + "/status")
    if response.json()["alive"]:
        print ("Linked")