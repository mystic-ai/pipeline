import typing as t
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image

from pipeline import Pipeline, Variable, entity, pipe
from pipeline.objects import File
from pipeline.objects.graph import InputField, InputSchema


class ModelKwargs(InputSchema):
    num_images_per_prompt: int | None = InputField(
        title="num_images_per_prompt",
        description="The number of images to generate per prompt.",
        default=1,
        optional=True,
    )
    height: int | None = InputField(
        title="height",
        description="The height in pixels of the generated image.",
        default=512,
        optional=True,
        multiple_of=64,
        ge=64,
    )
    width: int | None = InputField(
        title="width",
        description="The width in pixels of the generated image.",
        default=512,
        optional=True,
        multiple_of=64,
        ge=64,
    )
    num_inference_steps: int | None = InputField(
        title="num_inference_steps",
        description=(
            "The number of denoising steps. More denoising steps "
            "usually lead to a higher quality image at the expense "
            "of slower inference."
        ),
        default=50,
        optional=True,
    )

    guidance_scale: float | None = InputField(
        title="guidance_scale",
        description="The guidance scale.",
        default=1.0,
        optional=True,
    )


# Put your model inside of the below entity class
@entity
class MyModelClass:
    @pipe(run_once=True, on_startup=True)
    def load(self) -> None:
        # Perform any operations needed to load your model here
        print("Loading model...", flush=True)

        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "lykon/dreamshaper-xl-v2-turbo", torch_dtype=torch.float16, variant="fp16"
        )
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.pipe = self.pipe.to(device)

        print("Model loaded!", flush=True)

    @pipe
    def predict(self, prompt: str, model_kwargs: ModelKwargs) -> t.List[File]:
        # Perform any operations needed to predict with your model here
        print("Predicting...")

        images = self.pipe(prompt, **model_kwargs.to_dict()).images

        print("Prediction complete!")
        output_images = []
        for i, image in enumerate(images):
            path = Path(f"/tmp/sd/image-{i}.jpg")
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(str(path))
            output_images.append(File(path=path, allow_out_of_context_creation=True))
        return output_images


with Pipeline() as builder:
    input_var = Variable(
        str,
        description="Input prompt",
        title="Input prompt",
    )
    model_kwargs = Variable(ModelKwargs)

    my_model = MyModelClass()
    my_model.load()

    output_var = my_model.predict(input_var, model_kwargs)

    builder.output(output_var)

my_new_pipeline = builder.get_pipeline()
