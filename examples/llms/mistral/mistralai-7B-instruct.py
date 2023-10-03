from typing import List

import torch
from transformers import pipeline

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
    max_length: int | None = InputField(default=100, ge=1, le=4096)


@entity
class LlamaPipeline:
    @pipe(on_startup=True, run_once=True)
    def load_model(self) -> None:
        self.pipe = pipeline(
            "text-generation",
            model="mistralai/Mistral-7B-Instruct-v0.1",
            device_map="auto",
            torch_dtype=torch.float16,
        )

    @pipe
    def inference(self, prompt: list, kwargs: ModelKwargs) -> List[str]:
        outputs = self.pipe(prompt, **kwargs.to_dict())

        return [
            output[0]["generated_text"].replace(_input, "").strip()
            for output, _input in zip(outputs, prompt)
        ]


with Pipeline() as builder:
    prompt = Variable(
        list,
        default=["[INST] <<SYS>> answer any question <</SYS>> What is love? [/INST]"],
    )
    kwargs = Variable(ModelKwargs)

    _pipeline = LlamaPipeline()
    _pipeline.load_model()
    out = _pipeline.inference(prompt, kwargs)

    builder.output(out)


my_pipeline = builder.get_pipeline()


environments.create_environment(
    "mistralai/mistral-7b",
    python_requirements=[
        "torch==2.0.1",
        "git+https://github.com/huggingface/transformers@211f93aab95d1c683494e61c3cf8ff10e1f5d6b7",  # noqa
        "diffusers==0.19.3",
        "accelerate==0.21.0",
    ],
    allow_existing=True,
)

# Upload
result = upload_pipeline(
    my_pipeline,
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/mistral-7b",
    minimum_cache_number=1,
    required_gpu_vram_mb=35_000,
    accelerators=[
        Accelerator.nvidia_a100,
    ],
)

print(f"Pipeline ID: {result.id}")

output = run_pipeline(
    result.id,
    "Hello, how are you?",
    ModelKwargs(),
)

print(output)
