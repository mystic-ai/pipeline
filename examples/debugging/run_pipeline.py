from pipeline.cloud.pipelines import run_pipeline
from pipeline.configuration import current_configuration

current_configuration.set_debug_mode(True)

output = run_pipeline(
    "meta/llama2-70B-chat:latest",
    [
        [
            {
                "role": "system",
                "content": "Reply with only a JSON, like {\"category\": \"{categories}\"}. Classify the tweet into one of the following categories: 'Social media', 'Cats', 'Technology', 'Politics', 'Raves', 'Lifestyle'.",  # noqa
            },
            {
                "role": "user",
                "content": "I love the new Meta Llama 2 model it's really good, yay Zuck!",  # noqa
            },
        ]
    ],
    {},
)
