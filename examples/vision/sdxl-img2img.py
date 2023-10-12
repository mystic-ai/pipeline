from pathlib import Path
from typing import List

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

from pipeline import Pipeline, Variable, entity, pipe
from pipeline.cloud import compute_requirements, environments, pipelines
from pipeline.objects import File
from pipeline.objects.graph import InputField, InputSchema


class ModelKwargs(InputSchema):
    num_inference_steps: int | None = InputField(
        title="Number of inference steps",
        default=25,
    )
    strength: float | None = InputField(
        title="Strength",
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "The strength of the input image, 1.0 means the input "
            "image is fully ignored, 0.0 means the input image is fully used."
        ),
    )


@entity
class StableDiffusionModel:
    @pipe(on_startup=True, run_once=True)
    def load(self):
        self.base = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        self.base.to("cuda")

    @pipe
    def predict(
        self, prompt: str, input_image: File, kwargs: ModelKwargs
    ) -> List[File]:
        input_image_loaded = load_image(str(input_image.path)).convert("RGB")
        largest_side = max(input_image_loaded.size)
        if largest_side > 1024:
            input_image_loaded = input_image_loaded.resize(
                size=(
                    int(input_image_loaded.size[0] / largest_side * 1024),
                    int(input_image_loaded.size[1] / largest_side * 1024),
                ),
            )

        image = self.base(
            prompt=prompt,
            image=input_image_loaded,
            num_inference_steps=kwargs.num_inference_steps,
            strength=kwargs.strength,
            original_size=input_image_loaded.size,
            target_size=input_image_loaded.size,
        ).images

        output_images = []
        for i, image in enumerate(image):
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

    input_image = Variable(
        File,
        title="Input image",
    )

    kwargs = Variable(
        ModelKwargs,
        title="Model kwargs",
    )

    model = StableDiffusionModel()

    model.load()

    output = model.predict(prompt, input_image, kwargs)

    builder.output(output)

my_pl = builder.get_pipeline()

try:
    environments.create_environment(
        "stabilityai/sdxl-img2img",
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
    "stabilityai/sdxl-img2img",
    environment_id_or_name="stabilityai/sdxl-img2img",
    required_gpu_vram_mb=35_000,
    accelerators=[
        compute_requirements.Accelerator.nvidia_a100,
    ],
)

print(remote_pipeline.id)

output = my_pl.run(
    "sketch of a japanese spitz",
    File(
        path="image-1.jpeg",
    ),
    ModelKwargs(num_inference_steps=25),
)

print(output)
