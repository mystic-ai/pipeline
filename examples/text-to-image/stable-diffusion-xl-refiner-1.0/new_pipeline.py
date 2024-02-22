from pathlib import Path

import torch

# from diffusers import StableDiffusionXLImg2ImgPipeline
# from diffusers import StableDiffusionXLPipeline
from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
from diffusers.utils import load_image
from PIL import Image

from pipeline import File, Pipeline, Variable, entity, pipe
from pipeline.objects.graph import InputField, InputSchema


class ModelKwargs(InputSchema):
    num_images_per_prompt: int = InputField(
        title="num_images_per_prompt",
        description="The number of images to generate per prompt.",
        default=1,
        optional=True,
    )
    height: int = InputField(
        title="height",
        description="The height in pixels of the generated image.",
        default=512,
        optional=True,
        multiple_of=64,
        ge=64,
    )
    width: int = InputField(
        title="width",
        description="The width in pixels of the generated image.",
        default=512,
        optional=True,
        multiple_of=64,
        ge=64,
    )
    num_inference_steps: int = InputField(
        title="num_inference_steps",
        description=(
            "The number of denoising steps. More denoising steps "
            "usually lead to a higher quality image at the expense "
            "of slower inference."
        ),
        default=4,
        optional=True,
    )

    source_image: File | None = InputField(
        title="source_image",
        description="The source image to condition the generation on.",
        optional=True,
        default=None,
    )

    strength: float | None = InputField(
        title="strength",
        description="The strength of the new image from the input image. The lower the strength the closer to the original image the output will be.",  # noqa
        default=0.8,
        optional=True,
        ge=0,
        le=1,
    )


# Put your model inside of the below entity class
@entity
class MyModelClass:
    @pipe(run_once=True, on_startup=True)
    def load(self) -> None:
        # Perform any operations needed to load your model here
        print("Loading model...")

        self.pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
        )
        self.pipeline_text2image = self.pipeline_text2image.to("cuda")
        self.pipeline_image2image = AutoPipelineForImage2Image.from_pipe(
            self.pipeline_text2image
        )

        print("Model loaded!")

    @pipe
    def predict(self, prompt: str, kwargs: ModelKwargs) -> list[File]:
        # Perform any operations needed to predict with your model here
        print("Predicting...")
        if not hasattr(self, "pipeline_text2image"):
            raise ValueError("Model not loaded")

        if kwargs.source_image is not None:
            if kwargs.num_inference_steps * kwargs.strength < 1:
                raise ValueError(
                    "The strength and the number of inference steps are too low."
                    "Please increase the number of inference steps or the strength so that the product is at least 1."  # noqa
                )

            source_image = kwargs.source_image
            image = load_image(str(source_image.path))
            new_width = kwargs.width
            new_height = kwargs.height

            # # Calculate the center crop box
            # img_width, img_height = image.size
            # left = (img_width - new_width) / 2
            # top = (img_height - new_height) / 2
            # right = (img_width + new_width) / 2
            # bottom = (img_height + new_height) / 2
            # box = (left, top, right, bottom)
            # print(box)

            # # Perform the crop
            # cropped_img = image.crop(box)

            # resized_img = cropped_img.resize((new_width, new_height), Image.BICUBIC)

            # Calculate the new dimensions while preserving aspect ratio
            img_width, img_height = image.size
            aspect_ratio = img_width / img_height
            if img_width > img_height:
                new_height = int(new_width / aspect_ratio)
            else:
                new_width = int(new_height * aspect_ratio)

            # Resize the image
            img = image.resize((new_width, new_height), Image.LANCZOS)

            # Calculate the center crop box
            left = (new_width - new_height) / 2
            top = (new_height - new_width) / 2
            right = (new_width + new_height) / 2
            bottom = (new_height + new_width) / 2
            box = (left, top, right, bottom)

            # Perform the crop
            cropped_img = img.crop(box)

            input_kwargs = kwargs.to_dict()
            input_kwargs.pop("source_image")
            # input_kwargs.pop("height")
            # input_kwargs.pop("width")

            images = self.pipeline_image2image(
                image=cropped_img,
                prompt=prompt,
                guidance_scale=0.0,
                **input_kwargs,
            ).images

        else:
            images = self.pipeline_text2image(
                prompt=prompt,
                guidance_scale=0.0,
                **kwargs.to_dict(),
            ).images
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

    kwargs = Variable(
        ModelKwargs,
        description="Model arguments",
        title="Model arguments",
    )

    my_model = MyModelClass()
    my_model.load()

    output_var = my_model.predict(input_var, kwargs)

    builder.output(output_var)

my_new_pipeline = builder.get_pipeline()
