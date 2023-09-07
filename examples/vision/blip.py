from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from pipeline import File, Pipeline, Variable, entity, pipe
from pipeline.cloud import compute_requirements, pipelines
from pipeline.cloud.environments import create_environment


@entity
class BlipModel:
    def __init__(self):
        ...

    @pipe(on_startup=True, run_once=True)
    def load(self) -> None:
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )

    @pipe
    def predict(self, image: File) -> str:
        raw_image = Image.open(image.path.read_bytes()).convert("RGB")
        inputs = self.processor(raw_image, return_tensors="pt")
        out = self.model.generate(**inputs)
        return str(self.processor.decode(out[0], skip_special_tokens=True))


with Pipeline() as builder:
    image = Variable(File, title="Image File")

    model = BlipModel()

    model.load()

    output = model.predict(image)

    builder.output(output)

my_pl = builder.get_pipeline()

env_name = "blip"
try:
    env_id = create_environment(
        name=env_name,
        python_requirements=[
            "torch==2.0.1",
            "requests==2.30.0",
            "transformers==4.32.0",
            "pillow==10.0.0",
        ],
    )
except Exception:
    pass

pipelines.upload_pipeline(
    my_pl,
    "blip",
    environment_id_or_name=env_name,
    required_gpu_vram_mb=10_000,
    accelerators=[
        compute_requirements.Accelerator.nvidia_l4,
    ],
)
