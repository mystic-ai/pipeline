"""
Pipline has support for creating FastAPI endpoints for your Pipelines out of the box.
This file demonstrates this by creating a basic Stable Diffusion Pipeline with a custom
environment and deploying the docker image.

Run this code in this directory:

pip install -U "transformers==4.21.2" "tokenizers==0.12.1" \
"torch" \
"diffusers @ git+https://github.com/huggingface/
diffusers.git@5755d16868ec3da7d5eb4f42db77b01fac842ea8"

env HF_TOKEN="your hugging face token" python stable_diffusion_2_0_docker.py

and then start the docker containers created:

sudo docker compose up -d

you can test the running images:

curl -L -X POST 'http://localhost:5010/v2/runs' \
-H 'Content-Type: application/json' \
-d '{
    "pipeline_id": "sd-2_0",
    "data": [
        [
            {
                "text_in": "Mountain winds, and babbling springs, and moonlight seas"
            }
        ],
        {
            "num_samples": 1,
            "num_inference_steps": 25,
            "width": 768,
            "height": 768
        }
    ]
}'
"""

import os
import random
from typing import Optional, TypedDict

import numpy as np
import torch
from cloudpickle import dumps
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from dill import loads

from pipeline import (
    Pipeline,
    PipelineFile,
    Variable,
    docker,
    pipeline_function,
    pipeline_model,
)
from pipeline.objects.environment import Environment

scheduler = DPMSolverMultistepScheduler.from_pretrained(
    "stabilityai/stable-diffusion-2", subfolder="scheduler"
)
model = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2",
    use_auth_token=os.environ["HF_TOKEN"],
    scheduler=scheduler,
    revision="fp16",
    torch_dtype=torch.float16,
    safety_checker=None,
)

temp_path = "./temporary.model"
with open(temp_path, "wb") as tmp_file:
    tmp_file.write(dumps(model))

#
# pipeline.ai logic
#


def seed_everything(seed: int) -> int:
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def get_aspect_ratio(width: int, height: int) -> float:
    return width / height


class PromptShape(TypedDict):
    text_in: str
    seed: Optional[int]


class BatchKwargsShape(TypedDict):
    num_samples: Optional[int]
    height: Optional[int]
    width: Optional[int]
    seed: Optional[int]
    num_inference_steps: Optional[int]
    guidance_scale: Optional[float]
    eta: Optional[float]
    randomise_seed: Optional[bool]


@pipeline_model
class StableDiffusionTxt2ImgModel:
    @pipeline_function
    def predict(
        self, prompts: list[PromptShape], batch_kwargs: BatchKwargsShape
    ) -> list[list[str]]:
        print("🦁 run start")
        print(prompts)
        import base64
        import random
        from io import BytesIO

        default_batch_kwargs = {
            "num_samples": 1,
            "num_inference_steps": 25,
            "guidance_scale": 7.5,
            "eta": 0.0,
            "randomise_seed": True,
            "width": 768,
            "height": 768,
        }
        kwargs = {**default_batch_kwargs, **batch_kwargs}

        if not isinstance(kwargs["num_samples"], int):
            raise TypeError("num_samples must be an integer.")
        if not isinstance(kwargs.get("width", 0), int):
            raise TypeError("width must be an integer because half-pixels don't exist.")
        if not isinstance(kwargs.get("height", 1), int):
            raise TypeError(
                "height must be an integer because half-pixels don't exist."
            )
        if not isinstance(kwargs.get("seed", 1), int):
            raise TypeError("seed must be an integer.")
        if not isinstance(kwargs["num_inference_steps"], int):
            raise TypeError(
                "num_inference_steps must be an integer because denoising is done in \
                full non-fractional steps."
            )
        if kwargs["num_samples"] > 4:
            raise ValueError(
                "num_samples must be less than 4 in this version of the pipeline."
            )
        if kwargs.get("width", 1) < 1:
            raise ValueError("width can't be negative or 0.")
        if kwargs.get("height", 1) < 1:
            raise ValueError("height can't be negative or 0.")
        if kwargs["num_inference_steps"] < 1:
            raise ValueError("num_inference_steps can't be negative or 0.")

        base_seed_if_not_randomised = random.randint(1, 1000000)

        all_outputs = []
        for index, prompt in enumerate(prompts):
            print(f"🌀 prompt {index + 1}/{len(prompts)}: {prompt['text_in']}")

            if "seed" in prompt:
                seed_everything(prompt["seed"])
            elif "seed" in kwargs:
                seed_everything(kwargs["seed"])
                prompt["seed"] = kwargs["seed"]
            elif kwargs["randomise_seed"]:
                random_seed = random.randint(1, 1000000)
                seed_everything(random_seed)
                prompt["seed"] = random_seed

            metadata = {
                "scheduler": "multistep_dpm_solver",
                "seed": prompt["seed"]
                if "seed" in prompt
                else kwargs["seed"]
                if "seed" in kwargs
                else random_seed
                if kwargs["randomise_seed"]
                else base_seed_if_not_randomised,
            }

            generator = torch.Generator(device=self.device).manual_seed(prompt["seed"])

            prompt_images = []

            images = self.model(
                prompt=prompt["text_in"],
                guidance_scale=kwargs["guidance_scale"],
                generator=generator,
                num_images_per_prompt=kwargs["num_samples"],
                num_inference_steps=kwargs["num_inference_steps"],
                eta=kwargs["eta"],
                width=kwargs["width"],
                height=kwargs["height"],
            ).images

            for image in images:
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                prompt_images.append(img_str)

            if kwargs["guidance_scale"] == 1:
                metadata["classifier_free_guidance"] = False
            else:
                metadata["classifier_free_guidance"] = True

            prompt_dict = {"images_out": prompt_images, "metadata": metadata}

            all_outputs.append(prompt_dict)

            print(f"✅ prompt {index}/{len(prompts) - 1}")

        print("🐸 run complete")
        return all_outputs

    @pipeline_function(run_once=True, on_startup=True)
    def load(self, model_file: PipelineFile) -> bool:

        # it would be lovely to pass `device` to this load function, but for now...
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.device = device

        print(f"🔋 loading model (device:{self.device})")

        with open(model_file.path, "rb") as tmp_file:
            self.model = loads(tmp_file.read())
        self.model.to(device)

        print("🥑 model and weights load complete")

        return True


with Pipeline("sd-2_0", min_gpu_vram_mb=11577) as builder:
    model_file = PipelineFile(path="./temporary.model")
    prompts = Variable(list, is_input=True)
    batch_kwargs = Variable(dict, is_input=True)

    builder.add_variables(model_file, prompts, batch_kwargs)

    stable_diff_model = StableDiffusionTxt2ImgModel()
    stable_diff_model.load(model_file)

    output = stable_diff_model.predict(prompts, batch_kwargs)
    builder.output(output)

sd_pipeline = Pipeline.get_pipeline("sd-2_0")

env = Environment(
    dependencies=[
        "transformers==4.21.2",
        "tokenizers==0.12.1",
        "torch @ https://download.pytorch.org/whl/cu113/torch-1.12.0%"
        "2Bcu113-cp39-cp39-linux_x86_64.whl",
        "diffusers @ git+https://github.com/huggingface/diffusers.git"
        "@5755d16868ec3da7d5eb4f42db77b01fac842ea8",
    ]
)

docker.create_pipeline_api([sd_pipeline], gpu_index="0", environment=env)
docker.build()

os.remove("./temporary.model")
