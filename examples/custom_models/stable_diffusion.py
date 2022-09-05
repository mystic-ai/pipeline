import os

import torch

from cloudpickle import dumps
from pipeline import (
    pipeline_model,
    pipeline_function,
    Pipeline,
    PipelineCloud,
    PipelineFile,
    Variable,
)

from diffusers import StableDiffusionPipeline

# get your token at https://huggingface.co/settings/tokens
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=os.environ.get("HF_TOKEN"),
    revision="fp16",
    torch_dtype=torch.float16,
)


# Write the pipeline to a local file
temp_path = "tmp.pipeline"
with open(temp_path, "wb") as tmp_file:
    tmp_file.write(dumps(pipe))


@pipeline_model
class StableDiffusionModel:

    model = None

    def __init__(self):
        ...

    @pipeline_function
    def predict(self, x: list[str], input_kwargs: dict) -> list[str]:

        import random
        import base64

        from io import BytesIO

        results = []

        default_kwargs = {
            "seed": random.randint(1, 10000),
            "num_steps": 50,
            "diversity": 7.5,
            "width": 512,
            "height": 512,
            "eta": 0.0,
            "num_samples": 1,
        }

        kwargs = {**default_kwargs, **input_kwargs}
        generator = torch.Generator(self.device).manual_seed(
            default_kwargs.get("seed", 0)
        )
        if kwargs.get("num_samples", 1) > 1:
            new_prompts = []
            for _str in x:
                for _ in range(kwargs.get("num_samples", 1)):
                    new_prompts.append(_str)
            x = new_prompts

        if default_kwargs.get("width") % 8 != 0 or default_kwargs.get("height") % 8 != 0:
            return ["Error, width and height must be a multiple of 8"]

        for _str in x:
            with torch.autocast(self.device):
                res = self.model(
                    _str,
                    guidance_scale=kwargs["diversity"],
                    num_inference_steps=kwargs["num_steps"],
                    width=kwargs["width"],
                    height=kwargs["height"],
                    eta=kwargs["eta"],
                    generator=generator,
                )["sample"][0]
                buffered = BytesIO()
                res.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                results.append(img_str)

        return results

    @pipeline_function(run_once=True, on_startup=True)
    def load(self, model_file: PipelineFile) -> bool:
        from dill import loads
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading model...")
        with open(model_file.path, "rb") as tmp_file:
            self.model = loads(tmp_file.read())
        self.model.to(self.device)
        print("Loaded model!")

        return True


with Pipeline("stable_diffusion_pl") as builder:
    input_strs = Variable(list, is_input=True)
    input_kwargs = Variable(dict, is_input=True)
    model_pipeline_file = PipelineFile(path=temp_path)

    builder.add_variables(input_strs, input_kwargs, model_pipeline_file)

    stable_diff_model = StableDiffusionModel()
    stable_diff_model.load(model_pipeline_file)

    output = stable_diff_model.predict(input_strs, input_kwargs)

    builder.output(output)


test_pipeline = Pipeline.get_pipeline("stable_diffusion_pl")
# test_pipeline.run(["A small cat"],{})
# exit()
api = PipelineCloud()
uploaded_pipeline = api.upload_pipeline(test_pipeline)
print(f"Uploaded pipeline id: {uploaded_pipeline.id}")

# Cleanup local dir
os.remove(temp_path)
