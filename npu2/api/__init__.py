import requests
import inspect

from dill import dumps

from requests.api import head

from npu2.pipeline import Pipeline

#global API_ENDPOINT
#global API_TOKEN

API_ENDPOINT = "http://localhost:5002/v2"
API_TOKEN = ""

ALLOWED_UPLOADS = [Pipeline]

from npu2.api.link import link
from npu2.api.upload import upload