from pipeline import Pipeline, Variable, pipe
from pipeline.objects.graph import InputField, InputSchema


# Useful for testing out the frontend play form
class ModelParams(InputSchema):
    default_system_prompt: str = InputField(
        title="default_system_prompt",
        default="You are a helpful assistant.\n",
    )
    do_sample: bool = InputField(title="do_sample", default=False)
    max_new_tokens: int = InputField(title="max_new_tokens", default=100, ge=1, le=4096)
    presence_penalty: float = InputField(title="presence_penalty", default=1)
    temperature: float = InputField(title="temperature", default=0.6)
    top_k: float = InputField(title="top_k", default=50)
    top_p: float = InputField(title="top_p", default=0.9)
    use_cache: bool = InputField(title="use_cache", default=True)


@pipe
def predict(prompt: str) -> str:
    return f"{prompt} lol"


with Pipeline() as builder:
    prompt = Variable(str, max_length=64)
    params = Variable(ModelParams)
    output = predict(prompt)
    builder.output(output)

pipeline_graph = builder.get_pipeline()
