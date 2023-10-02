# flake8: noqa
from typing import List

from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams

from pipeline import Pipeline, entity, pipe
from pipeline.cloud import environments
from pipeline.cloud.compute_requirements import Accelerator
from pipeline.cloud.pipelines import run_pipeline, upload_pipeline
from pipeline.configuration import current_configuration
from pipeline.objects.graph import InputField, InputSchema, Variable

current_configuration.set_debug_mode(True)


class ModelKwargs(InputSchema):
    do_sample: bool | None = InputField(default=False)
    use_cache: bool | None = InputField(default=True)
    temperature: float | None = InputField(default=0.6)
    top_k: float | None = InputField(default=50)
    top_p: float | None = InputField(default=0.9)
    max_new_tokens: int | None = InputField(default=100, ge=1, le=1000)
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

        model_dir = Path("~/.cache/huggingface/llama2/7b-chat").expanduser()
        model_dir.mkdir(parents=True, exist_ok=True)
        model_dir = str(model_dir)

        snapshot_download(
            "meta-llama/Llama-2-7b-chat-hf",
            local_dir=model_dir,
            token="",
            # force_download=True,
            ignore_patterns=["*.safetensors"],
        )
        self.llm = LLM(
            model_dir,
            dtype="bfloat16",
        )
        self.tokenizer = self.llm.get_tokenizer()

    @pipe
    def inference(self, prompts: list, kwargs: ModelKwargs) -> List[str]:
        sampling_params = SamplingParams(
            temperature=kwargs.temperature,
            top_p=kwargs.top_p,
            max_tokens=kwargs.max_new_tokens,
            presence_penalty=kwargs.presence_penalty,
        )

        generation_tokens = self.llm.generate(
            prompts,
            sampling_params,
        )

        return [t.outputs[0].text for t in generation_tokens]


with Pipeline() as builder:
    prompt = Variable(list)
    kwargs = Variable(ModelKwargs)

    _pipeline = LlamaPipeline()
    _pipeline.load_model()
    out = _pipeline.inference(prompt, kwargs)

    builder.output(out)


my_pipeline = builder.get_pipeline()

environments.create_environment(
    "meta/llama2-vllm",
    python_requirements=[
        "torch==2.0.1",
        "transformers==4.30.2",
        "diffusers==0.19.3",
        "accelerate==0.21.0",
        "hf-transfer~=0.1",
        "vllm==0.1.4",
    ],
)

# Upload
result = upload_pipeline(
    my_pipeline,
    "meta/llama2-7B-instruct",
    "meta/llama2-vllm",
    required_gpu_vram_mb=35_000,
    accelerators=[
        Accelerator.nvidia_a100,
    ],
)
run_pipeline(
    result.id,
    [
        "[INST] <<SYS>> You are a genius 10x programmer <</SYS>> How do I install Python packages? [/INST]",
        "[INST] <<SYS>> IDIOT advisor <</SYS>> How do I install Python packages? [/INST]",
    ],
    {},
)
