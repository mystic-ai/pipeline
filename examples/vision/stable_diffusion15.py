import torch
from diffusers import StableDiffusionPipeline

from pipeline import Pipeline, Variable, pipe, pipeline_model


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
    def predict(self, prompt: str, kwargs: dict) -> list:
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

        import base64
        from io import BytesIO

        output_images = []
        for image in images:
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue())
            output_images.append(img_str.decode("utf-8"))

        return output_images


with Pipeline() as builder:
    prompt = Variable(str)
    kwargs = Variable(dict)

    model = StableDiffusionModel()

    model.load()

    output = model.predict(prompt, kwargs)

    builder.output(output)

my_pl = builder.get_pipeline()

print(
    my_pl.run(
        "A dog",
        {
            "num_images_per_prompt": 4,
        },
    )
)
