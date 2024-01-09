from typing import List

from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams

from pipeline import Pipeline, entity, pipe
from pipeline.configuration import current_configuration
from pipeline.objects.graph import InputField, InputSchema, Variable

current_configuration.set_debug_mode(True)


class ModelKwargs(InputSchema):
    do_sample: bool | None = InputField(default=False)
    use_cache: bool | None = InputField(default=True)
    temperature: float | None = InputField(default=0.6)
    top_k: float | None = InputField(default=50)
    top_p: float | None = InputField(default=0.9)
    max_new_tokens: int | None = InputField(default=100, ge=1, le=4096)
    presence_penalty: float | None = InputField(default=1.0)


@entity
class LlamaPipeline:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None

        self.streamer = None

    @pipe(on_startup=True, run_once=True)
    def load_model(self) -> None:
        from pathlib import Path

        model_dir = Path("~/.cache/huggingface/llama2/13b").expanduser()
        model_dir.mkdir(parents=True, exist_ok=True)
        model_dir = str(model_dir)
        snapshot_download(
            "meta-llama/Llama-2-13b-hf",
            local_dir=model_dir,
            token="",
            force_download=True,
        )
        self.llm = LLM(
            model_dir,
            dtype="bfloat16",
        )

    @pipe
    def inference(self, prompt: str, kwargs: ModelKwargs) -> List[str]:
        prompts = prompt
        if isinstance(prompts, str):
            prompts = [prompts]

        sampling_params = SamplingParams(
            temperature=kwargs.temperature,
            top_p=kwargs.top_p,
            max_tokens=kwargs.max_new_tokens,
            presence_penalty=kwargs.presence_penalty,
        )

        result = self.llm.generate(prompts, sampling_params)

        return result[0].outputs[0].text


with Pipeline() as builder:
    prompt = Variable(str)
    kwargs = Variable(ModelKwargs)

    _pipeline = LlamaPipeline()
    _pipeline.load_model()
    out = _pipeline.inference(prompt, kwargs)

    builder.output(out)


my_new_pipeline = builder.get_pipeline()
