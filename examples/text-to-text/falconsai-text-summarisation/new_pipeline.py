import typing as t

from transformers import pipeline as hf_pipeline

from pipeline import Pipeline, Variable, entity, pipe
from pipeline.objects.graph import File, InputField, InputSchema

HF_MODEL_NAME = "Falconsai/text_summarization"


class Inputs(InputSchema):
    text_in: str | None = InputField(
        default=None,
        optional=True,
        title="raw_text",
        description="Raw text to be summarised",
    )
    file_in: File | None = InputField(
        default=None,
        optional=True,
        title="text_file",
        description="a .txt file to be summarised",
    )


class ModelKwargs(InputSchema):
    max_length: int | None = InputField(default=230, optional=True)
    min_length: int | None = InputField(default=30, optional=True)
    do_sample: bool | None = InputField(default=False)


@entity
class Summarizer:
    def __init__(self):
        self.summariser = None

    @pipe
    def _read_text(self, file_in: File) -> str:
        with open(file_in.path, "r") as f:
            return f.read()

    @pipe
    def resolve_text_to_summarise(self, inputs: Inputs) -> str:
        if not inputs.file_in and not inputs.text_in:
            raise Exception("Must either input raw text or a file")
        if inputs.file_in is not None:
            return self._read_text(inputs.file_in)
        return inputs.text_in

    @pipe(run_once=True, on_startup=True)
    def load(self) -> None:
        self.summarizer = hf_pipeline("summarization", model=HF_MODEL_NAME)

    @pipe
    def predict(self, text: str, model_kwargs: ModelKwargs) -> t.Any:
        summary = self.summarizer(text, **model_kwargs.to_dict())
        return summary


with Pipeline() as builder:
    inputs = Variable(Inputs, description="Either enter raw text or upload a file")
    model_kwargs = Variable(ModelKwargs)

    model = Summarizer()
    model.load()
    text = model.resolve_text_to_summarise(inputs)

    summary = model.predict(text, model_kwargs)

    builder.output(summary)

my_new_pipeline = builder.get_pipeline()
