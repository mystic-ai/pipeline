from typing import List

from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams

from pipeline import Pipeline, entity, pipe
from pipeline.cloud import environments
from pipeline.cloud.compute_requirements import Accelerator
from pipeline.cloud.pipelines import upload_pipeline
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

        model_dir = Path("~/.cache/huggingface/llama2/70b").expanduser()
        model_dir.mkdir(parents=True, exist_ok=True)
        model_dir = str(model_dir)
        snapshot_download(
            "meta-llama/Llama-2-70b-hf",
            local_dir=model_dir,
            token="",
            force_download=True,
            ignore_patterns=["*.safetensors"],
        )

        self.llm = LLM(
            model_dir,
            dtype="bfloat16",
            tensor_parallel_size=2,
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


my_pipeline = builder.get_pipeline()


my_env = environments.create_environment(
    "meta/llama2-vllm-ray",
    python_requirements=[
        "torch==2.0.1",
        "transformers",
        "diffusers==0.19.3",
        "accelerate==0.21.0",
        "hf-transfer~=0.1",
        "vllm==0.1.4",
        "ray==2.6.3",
        "pandas",  # for ray - will say ray is not installed otherwise
    ],
)

# Upload
result = upload_pipeline(
    my_pipeline,
    "meta/llama2-70B",
    "meta/llama2-vllm-ray",
    required_gpu_vram_mb=140_000,
    accelerators=[
        Accelerator.nvidia_a100_80gb,
        Accelerator.nvidia_a100_80gb,
    ],
)
