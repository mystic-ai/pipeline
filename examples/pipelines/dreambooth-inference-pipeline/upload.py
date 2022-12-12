import os
from cloudpickle import dumps
from dill import loads
from typing import TypedDict, Optional
import torch
import numpy as np
import random

from pipeline import (
    pipeline_function,
    pipeline_model,
    Pipeline,
    PipelineCloud,
    Variable,
    PipelineFile,
)

from diffusers import DiffusionPipeline
from diffusers.utils import logging

logging.disable_progress_bar()
logging.set_verbosity_error()


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


PIPELINE_NAME = "sd-dreambooth-library/herge-style"
# Download dreambooth pre-trained pipeline from HF, but
# feel free to use your own pre-trained pipeline or pytorch model here instead
sd_pipeline = DiffusionPipeline.from_pretrained(PIPELINE_NAME)


temp_path = "temporary.model"
with open(temp_path, "wb") as tmp_file:
    tmp_file.write(dumps(sd_pipeline))


def seed_everything(seed: int) -> int:
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


@pipeline_model
class StableDiffusionTxt2ImgModel:
    @pipeline_function
    def predict(
        self, prompts: list[PromptShape], batch_kwargs: BatchKwargsShape
    ) -> list[list[str]]:
        from io import BytesIO
        import base64
        import random

        default_batch_kwargs = {
            "num_samples": 1,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "eta": 0.0,
            "randomise_seed": True,
            "width": 512,
            "height": 512,
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
                "num_inference_steps must be an integer because denoising is done in full non-fractional steps."
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
                "scheduler": "ddim",
                "seed": prompt["seed"]
                if "seed" in prompt
                else kwargs["seed"]
                if "seed" in kwargs
                else random_seed
                if kwargs["randomise_seed"]
                else base_seed_if_not_randomised,
            }

            generator = torch.Generator(device=0).manual_seed(prompt["seed"])

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

        return all_outputs

    @pipeline_function(run_once=True, on_startup=True)
    def load(self, model_file: PipelineFile) -> bool:

        device = torch.device("cuda:0")

        with open(model_file.path, "rb") as tmp_file:
            self.model = loads(tmp_file.read())
        self.model.to(device)

        return True


with Pipeline(PIPELINE_NAME, min_gpu_vram_mb=3040) as builder:
    model_file = PipelineFile(path="temporary.model")
    prompts = Variable(list, is_input=True)
    batch_kwargs = Variable(dict, is_input=True)

    builder.add_variables(model_file, prompts, batch_kwargs)

    stable_diff_model = StableDiffusionTxt2ImgModel()
    stable_diff_model.load(model_file)

    output = stable_diff_model.predict(prompts, batch_kwargs)
    builder.output(output)


new_pipeline = Pipeline.get_pipeline(PIPELINE_NAME)

api = PipelineCloud()
uploaded_pipeline = api.upload_pipeline(new_pipeline)
print(f"Uploaded pipeline id: {uploaded_pipeline.id}")

os.remove("temporary.model")
