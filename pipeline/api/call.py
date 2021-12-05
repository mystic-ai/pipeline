import inspect
import requests
import urllib.parse
from tqdm import tqdm

from requests_toolbelt.multipart import encoder

# from pipeline.api import PIPELINE_API_TOKEN, PIPELINE_API_URL
import pipeline.api
from pipeline.util.logging import PIPELINE_STR


def __handle_response__(response: requests.Response):
    if response.status_code == 404:
        raise Exception(response)


def post(endpoint, json_data):
    headers = {
        "Authorization": "Bearer %s" % pipeline.api.PIPELINE_API_TOKEN,
        "Content-type": "application/json",
    }

    url = urllib.parse.urljoin(pipeline.api.PIPELINE_API_URL, endpoint)
    response = requests.post(url, headers=headers, json=json_data)
    __handle_response__(response)
    return response.json()


def get(endpoint):
    headers = {"Authorization": "Bearer %s" % pipeline.api.PIPELINE_API_TOKEN}

    url = urllib.parse.urljoin(pipeline.api.PIPELINE_API_URL, endpoint)

    response = requests.get(url, headers=headers)
    __handle_response__(response)
    return response.json()


def post_file(endpoint, file, remote_path):
    url = urllib.parse.urljoin(pipeline.api.PIPELINE_API_URL, endpoint)
    e = encoder.MultipartEncoder(
        fields={
            "file_path": remote_path,
            "file": (
                file.name,
                file,
                "application/octet-stream",
                {"Content-Transfer-Encoding": "binary"},
            ),
        }
    )
    encoder_len = e.len
    bar = tqdm(
        desc=f"{PIPELINE_STR} Uploading",
        unit="B",
        unit_scale=True,
        total=encoder_len,
        unit_divisor=1024,
    )

    def progress_callback(monitor):
        bar.n = monitor.bytes_read
        bar.refresh()
        if monitor.bytes_read == encoder_len:
            bar.close()

    encoded_stream_data = encoder.MultipartEncoderMonitor(e, callback=progress_callback)

    headers = {
        "Authorization": "Bearer %s" % pipeline.api.PIPELINE_API_TOKEN,
        "Content-type": encoded_stream_data.content_type,
    }

    response = requests.post(url, headers=headers, data=encoded_stream_data)
    __handle_response__(response)
    return response.json()
