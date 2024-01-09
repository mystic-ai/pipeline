import typing as t
from pathlib import Path

from PIL.Image import Image
from pipeline.objects.graph import InputField, InputSchema

from pipeline import File, Pipeline, Variable, entity, pipe

HF_MODEL_ID = "runwayml/stable-diffusion-v1-5"


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


@entity
class StableDiffusionModel:
    def __init__(self) -> None:
        self.model = None
        self.device = None

    @pipe(run_once=True, on_startup=True)
    def load(self) -> None:
        """
        Load the HF model into memory"""
        import torch
        from diffusers import StableDiffusionPipeline

        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.model = StableDiffusionPipeline.from_pretrained(HF_MODEL_ID)
        self.model.to(device)

    @pipe
    def predict(self, prompt: str, model_kwargs: ModelKwargs) -> t.List[Image]:
        """
        Generates a list of PIL images.
        """
        return self.model(prompt=prompt, **model_kwargs.to_dict()).images

    @pipe
    def postprocess(self, images: t.List[Image]) -> t.List[File]:
        """
        Creates a list of Files from the `PIL` images.
        """
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
        title="prompt",
        description="The prompt to guide image generation",
        max_length=512,
    )
    model_kwargs = Variable(ModelKwargs)

    model = StableDiffusionModel()
    model.load()

    images: t.List[Image] = model.predict(prompt, model_kwargs)

    output: t.List[File] = model.postprocess(images)

    builder.output(output)

pipeline_graph = builder.get_pipeline()
