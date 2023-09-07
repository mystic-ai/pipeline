import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from pipeline import Pipeline, Variable, entity, pipe
from pipeline.cloud import compute_requirements, environments, pipelines
from pipeline.objects.graph import InputField, InputSchema


class ModelKwargs(InputSchema):
    max_length: int | None = InputField(default=512)
    padding: bool | None = InputField(default=True)
    truncation: bool | None = InputField(default=True)
    return_tensors: str | None = InputField(default="pt")


@entity
class E5Model:
    def __init__(self):
        ...

    @pipe(on_startup=True, run_once=True)
    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-large-v2")
        self.model = AutoModel.from_pretrained("intfloat/e5-large-v2")

    @pipe
    def predict(self, input_texts: list[str]) -> list:
        defaults = ModelKwargs().to_dict()

        batch_dict = self.tokenizer(input_texts, **defaults)
        outputs = self.model(**batch_dict)
        last_hidden_states = outputs.last_hidden_state
        attention_mask = batch_dict["attention_mask"]
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.tolist()


with Pipeline() as builder:
    input_texts = Variable(
        list, title="Input Texts", description="Text to embed, as an array of strings"
    )
    kwargs = Variable(ModelKwargs)

    model = E5Model()
    model.load()

    output = model.predict(input_texts, kwargs)

    builder.output(output)

my_pl = builder.get_pipeline()

env_name = "e5_large-v2"
try:
    environments.create_environment(
        name=env_name,
        python_requirements=[
            "torch==2.0.1",
            "sentence_transformers==2.2.2",
        ],
    )
except Exception:
    pass

pipelines.upload_pipeline(
    my_pl,
    "e5_large-v2",
    environment_id_or_name=env_name,
    required_gpu_vram_mb=10_000,
    accelerators=[
        compute_requirements.Accelerator.nvidia_l4,
    ],
)
