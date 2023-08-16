from pathlib import Path
from typing import List

import torch
from diffusers import StableDiffusionPipeline

from pipeline import Pipeline, Variable, pipe, pipeline_model
from pipeline.cloud import compute_requirements, environments, pipelines
from pipeline.objects import File


@pipeline_model
class StableDiffusionModel:
    def __init__(self):
        ...

    @pipe(on_startup=True, run_once=True)
    def load(self):
        model_id = "runwayml/stable-diffusion-v1-5"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
        )
        self.pipe = self.pipe.to(device)

    @pipe
    def predict(self, prompt: str, kwargs: dict) -> List[File]:
        defaults = {
            "height": 512,
            "width": 512,
            "num_inference_steps": 50,
            "num_images_per_prompt": 1,
            "guidance_scale": 7.5,
        }

        defaults.update(kwargs)
        defaults["output_type"] = "pil"
        images = self.pipe(prompt, **defaults).images

        output_images = []
        for i, image in enumerate(images):
            path = Path(f"/tmp/sd/image-{i}.jpg")
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(str(path))
            output_images.append(File(path=path, allow_out_of_context_creation=True))

        # import base64
        # from io import BytesIO

        # output_images = []
        # for image in images:
        #     buffered = BytesIO()
        #     image.save(buffered, format="JPEG")
        #     img_str = base64.b64encode(buffered.getvalue())
        #     output_images.append(img_str.decode("utf-8"))

        return output_images[0]


with Pipeline() as builder:
    prompt = Variable(str)
    kwargs = Variable(dict)

    model = StableDiffusionModel()

    model.load()

    output = model.predict(prompt, kwargs)

    builder.output(output)

my_pl = builder.get_pipeline()

try:
    environments.create_environment(
        "stable-diffusion",
        python_requirements=[
            "torch==2.0.1",
            "transformers==4.30.2",
            "diffusers==0.19.3",
            "accelerate==0.21.0",
        ],
    )
except Exception:
    pass


pipelines.upload_pipeline(
    my_pl,
    "stable-diffusion:latest",
    environment_id_or_name="stable-diffusion",
    required_gpu_vram_mb=10_000,
    accelerators=[
        compute_requirements.Accelerator.nvidia_a5000,
    ],
)
# print(
#     my_pl.run(
#         "A dog",
#         {
#             "num_images_per_prompt": 4,
#         },
#     )
# )
