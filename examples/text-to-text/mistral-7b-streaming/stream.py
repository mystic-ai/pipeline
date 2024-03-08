import json

import httpx

url = "http://localhost:14300/v4/runs"

json_ = {
    "inputs": [
        {
            "type": "array",
            "value": [
                "hey there, my name is paul and streaming is my bitch. lets talk about my cats..."  # noqa
            ],
        },
        {"type": "dictionary", "value": {"max_new_tokens": 1000}},
    ]
}

i = 0
with httpx.stream("POST", url, json=json_) as response:
    for chunk in response.iter_bytes():
        # print(chunk)
        json_data = json.loads(chunk.decode())
        # i += 1
        print(json_data["outputs"][0]["value"], end="")
        # if i == 5:
        #     break
