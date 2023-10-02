from typing import List

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
    max_tokens: int | None = InputField(default=100, ge=1, le=4096)


@entity
class Mistral7B:
    @pipe(on_startup=True, run_once=True)
    def load_model(self) -> None:
        self.llm = LLM("mistralai/Mistral-7B-v0.1")

    @pipe
    def inference(self, prompts: list, kwargs: ModelKwargs) -> List[str]:
        sampling_params = SamplingParams(
            temperature=kwargs.temperature,
            top_p=kwargs.top_p,
            max_tokens=kwargs.max_tokens,
        )

        result = self.llm.generate(prompts, sampling_params)

        return [t.outputs[0].text for t in result]


with Pipeline() as builder:
    prompt = Variable(list, default=["My name is"])
    kwargs = Variable(ModelKwargs)

    _pipeline = Mistral7B()
    _pipeline.load_model()
    out = _pipeline.inference(prompt, kwargs)

    builder.output(out)


my_pipeline = builder.get_pipeline()


environments.create_environment(
    "mistralai/mistral-7b-vllm",
    python_requirements=[
        "torch==2.0.1",
        "git+https://github.com/huggingface/transformers@211f93aab95d1c683494e61c3cf8ff10e1f5d6b7",  # noqa
        "diffusers==0.19.3",
        "accelerate==0.21.0",
        "vllm==0.2.0",
    ],
)

# Upload
result = upload_pipeline(
    my_pipeline,
    "mistralai/Mistral-7B-v0.1",
    "mistralai/mistral-7b-vllm",
    minimum_cache_number=1,
    required_gpu_vram_mb=35_000,
    accelerators=[
        Accelerator.nvidia_a100,
    ],
)

run_pipeline(
    result.id,
    ["Hello my name is"],
    ModelKwargs(),
)
