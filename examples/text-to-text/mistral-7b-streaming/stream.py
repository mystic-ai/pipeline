import httpx

from pipeline.util.streaming import handle_stream_response

url = "http://localhost:14300/v4/runs/stream"

json_ = {
    "inputs": [
        {
            "type": "array",
            "value": [
                "hey there, my name is paul and I like building robot cars. Let me tell you about the latest one I made."
            ],
        },
        {
            "type": "dictionary",
            "value": {
                "max_new_tokens": 500,
                "do_sample": True,
                "repetition_penalty": 1.5,
            },
        },
    ]
}

with httpx.stream("POST", url, json=json_) as response:
    for json_data in handle_stream_response(response):
        if json_data.get("outputs"):
            print(json_data["outputs"][0]["value"], end="")
        else:
            print(json_data)
