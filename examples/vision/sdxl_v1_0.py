from pathlib import Path
from typing import List

import torch
from diffusers import DiffusionPipeline

from pipeline import Pipeline, Variable, entity, pipe
from pipeline.cloud import compute_requirements, environments, pipelines
from pipeline.objects import File
from pipeline.objects.graph import InputField, InputSchema


class ModelKwargs(InputSchema):
    num_inference_steps: int | None = InputField(
        title="Number of inference steps",
        default=25,
    )

    denoising_end: float | None = InputField(
        default=0.8,
        title="Denoising end",
        ge=0.0,
        le=1.0,
    )


@entity
class StableDiffusionModel:
    @pipe(on_startup=True, run_once=True)
    def load(self):
        self.base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        self.base.to("cuda")
        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.refiner.to("cuda")

    @pipe
    def predict(self, prompt: str, kwargs: ModelKwargs) -> List[File]:
        image = self.base(
            prompt=prompt,
            num_inference_steps=kwargs.num_inference_steps,
            denoising_end=kwargs.denoising_end,
            output_type="latent",
        ).images
        images = self.refiner(
            prompt=prompt,
            num_inference_steps=kwargs.num_inference_steps,
            denoising_start=kwargs.denoising_end,
            image=image,
        ).images

        output_images = []
        for i, image in enumerate(images):
            path = Path(f"/tmp/sd/image-{i}.jpg")
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(str(path))
            output_images.append(File(path=path, allow_out_of_context_creation=True))

        return output_images


with Pipeline() as builder:
    prompt = Variable(
        str,
        title="Prompt",
    )
    kwargs = Variable(
        ModelKwargs,
        title="Model kwargs",
    )

    model = StableDiffusionModel()

    model.load()

    output = model.predict(prompt, kwargs)

    builder.output(output)

my_pl = builder.get_pipeline()

try:
    environments.create_environment(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        python_requirements=[
            "torch==2.0.1",
            "transformers==4.30.2",
            "diffusers==0.19.3",
            "accelerate==0.21.0",
            "xformers==0.0.21",
            "invisible_watermark==0.2.0",
            "safetensors==0.3.3",
        ],
    )
except Exception:
    pass


remote_pipeline = pipelines.upload_pipeline(
    my_pl,
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    environment_id_or_name="stabilityai/stable-diffusion-xl-refiner-1.0",
    required_gpu_vram_mb=20_000,
    accelerators=[
        compute_requirements.Accelerator.nvidia_a100,
    ],
)

print(remote_pipeline.id)
