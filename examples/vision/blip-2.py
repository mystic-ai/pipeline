from io import BytesIO

import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from pipeline import File, Pipeline, Variable, entity, pipe
from pipeline.cloud import compute_requirements, pipelines
from pipeline.cloud.environments import create_environment


@entity
class BlipModel:
    @pipe(on_startup=True, run_once=True)
    def load(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float32
        )
        self.model.to(self.device)

    @pipe
    def predict(self, image: File) -> str:
        raw_image = Image.open(BytesIO(image.path.read_bytes())).convert("RGB")
        inputs = self.processor(raw_image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        return str(self.processor.decode(out[0], skip_special_tokens=True))


with Pipeline() as builder:
    image = Variable(
        File,
        title="Image File",
        description="Upload a .png, .jpg or other image file to be captioned",
    )

    model = BlipModel()

    model.load()

    output = model.predict(image)

    builder.output(output)

my_pl = builder.get_pipeline()

env_name = "paulh/blip-2"

env_id = create_environment(
    name=env_name,
    python_requirements=[
        "torch==2.0.1",
        "requests==2.30.0",
        "transformers==4.32.0",
        "pillow==10.0.0",
    ],
)


pipelines.upload_pipeline(
    my_pl,
    "paulh/blip-2",
    environment_id_or_name=env_name,
    required_gpu_vram_mb=16_000,
    accelerators=[
        compute_requirements.Accelerator.nvidia_a100,
    ],
)
