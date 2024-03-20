from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as torch_transforms
from controlnet_aux.pidi import PidiNetDetector
from controlnet_aux.util import HWC3, resize_image
from diffusers import (
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
)
from PIL import Image

from pipeline import File, Pipeline, entity, pipe
from pipeline.objects.graph import InputField, InputSchema, Variable

style_list = [
    {
        "name": "(No style)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "photo, photorealistic, realism, ugly",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },
    {
        "name": "Pixel art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
        "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
        "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    },
    {
        "name": "Manga",
        "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style",
    },
]

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "(No style)"


class ModelKwargs(InputSchema):
    steps: Optional[int] = InputField(
        default=25,
        title="Temperature",
        description="Sampling temperature used for generation.",
    )
    guidance_scale: Optional[float] = InputField(
        default=5.0,
        title="Temperature",
        description="Sampling temperature used for generation.",
    )
    adapter_conditioning_scale: Optional[float] = InputField(
        default=0.8,
        title="Temperature",
        description="Sampling temperature used for generation.",
    )
    negative_prompt: Optional[str] = InputField(
        default="",
        title="Negative Prompt",
        description="Provide what you want the model to avoid generating.",
        optional=True,
    )
    style_name: Optional[str] = InputField(
        default=DEFAULT_STYLE_NAME,
        title="Style",
        description="Select a style to generate in.",
        choices=STYLE_NAMES,
    )


@entity
class DiffusionWithAdapter:
    def __init__(self) -> None: ...

    def apply_style(
        self, style_name: str, positive: str, negative: str = ""
    ) -> tuple[str, str]:
        p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
        return p.replace("{prompt}", positive), n + negative

    @pipe(on_startup=True, run_once=True)
    def load(self) -> None:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pidi = PidiNetDetector.from_pretrained("lllyasviel/Annotators").to(device)

        # load adapter
        adapter = T2IAdapter.from_pretrained(
            "TencentARC/t2i-adapter-sketch-sdxl-1.0",
            torch_dtype=torch.float16,
            varient="fp16",
        )

        # load euler_a scheduler
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"

        euler_a = EulerAncestralDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        )
        self.pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            model_id,
            vae=vae,
            adapter=adapter,
            scheduler=euler_a,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(device)
        self.pipe.enable_xformers_memory_efficient_attention()

    @pipe
    def inference(self, image: File, prompt: str, kwargs: ModelKwargs) -> list[File]:

        input_image = image
        image = Image.open(BytesIO(image.path.read_bytes())).convert("RGB")

        img_resolution_target = 768
        np_img = np.array(image)
        img = resize_image(HWC3(np_img), img_resolution_target)
        detected_map = np.zeros_like(img, dtype=np.uint8)
        detected_map[np.min(img, axis=2) < 80] = 255

        image = torch_transforms.to_tensor(detected_map) > 0.5
        image = torch_transforms.to_pil_image(image.to(torch.float32))

        prompt, negative_prompt = self.apply_style(
            kwargs.style_name, prompt, kwargs.negative_prompt
        )
        image = self.pidi(
            image, detect_resolution=1024, image_resolution=1024, apply_filter=True
        )

        gen_image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_inference_steps=kwargs.steps,
            adapter_conditioning_scale=kwargs.adapter_conditioning_scale,
            guidance_scale=kwargs.guidance_scale,
        ).images[0]

        print("Image generated")

        path = Path("/tmp/dif_w_adapter/image.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        gen_image.save(str(path))

        output_image = File(
            path=path, allow_out_of_context_creation=True
        )  # Return location of generated img
        return [input_image, output_image]


with Pipeline() as builder:

    image = Variable(
        File,
        title="Input sketch",
        description="Upload a .png, .jpg or other image file of a sketch",
    )
    prompt = Variable(str, title="Prompt", description="Prompt to generate from")
    kwargs = Variable(ModelKwargs)

    model = DiffusionWithAdapter()
    model.load()

    # Forward pass
    out = model.inference(image, prompt, kwargs)

    builder.output(out)

my_new_pipeline = builder.get_pipeline()
