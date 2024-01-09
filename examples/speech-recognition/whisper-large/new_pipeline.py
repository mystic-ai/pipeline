from typing import Tuple

import torch
from transformers import pipeline

from pipeline import Pipeline, Variable, entity, pipe
from pipeline.objects import File
from pipeline.objects.graph import InputField, InputSchema


class ModelKwargs(InputSchema):
    batch_size: int | None = InputField(
        default=8,
        ge=1,
        le=16,
        title="Batch Size",
    )

    return_timestamps: bool | None = InputField(
        default=False,
        title="Return Timestamps",
    )


@entity
class WhisperModel:
    def __init__(self):
        ...

    @pipe(on_startup=True, run_once=True)
    def load(self):
        self.device = torch.device(0 if torch.cuda.is_available() else "cpu")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v2",
            chunk_length_s=30,
            device=self.device,
        )

    @pipe
    def predict(self, audio_file: File, kwargs: ModelKwargs) -> Tuple[str, list | None]:
        prediction = self.pipe(
            str(audio_file.path),
            batch_size=kwargs.batch_size,
            return_timestamps=kwargs.return_timestamps,
        )

        full_text: str = prediction["text"]
        timestamps: list = prediction["chunks"] if kwargs.return_timestamps else None

        return (full_text, timestamps)


with Pipeline() as builder:
    audio_file = Variable(File)
    kwargs = Variable(ModelKwargs)

    model = WhisperModel()

    model.load()

    full_text, timestamps = model.predict(audio_file, kwargs)

    builder.output(full_text, timestamps)

my_new_pipeline = builder.get_pipeline()
