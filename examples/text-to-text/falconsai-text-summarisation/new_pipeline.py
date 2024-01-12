from pipeline import Pipeline, Variable, entity, pipe
from pipeline.objects.graph import InputSchema, InputField
from pipeline.cloud.schemas import BaseModel
from pydantic import validator

from transformers import pipeline as t_pipeline
import typing as t


HF_MODEL_NAME = "Falconsai/text_summarization"

class ModelKwargs(InputSchema):
    field1: str | None = InputField(default=None, optional=True)
    field2: t.Optional[str] = InputField(None)

    @InputSchema.validator('field1', 'field2')
    def check_fields(cls, field1, field2):
        print(dir(field1), field2) 
        if field1 is None and field2 is None:
            raise Exception("At least one field should be non-None.")


# Put your model inside of the below entity class
@entity
class Summarizer:
    def __init__(self):
        self.summariser = None

    @pipe(run_once=True, on_startup=True)
    def load(self) -> None:
        self.summarizer = t_pipeline("summarization", model=HF_MODEL_NAME)

    @pipe
    def predict(self, model_kwargs: ModelKwargs) -> t.Any:
        # Perform any operations needed to predict with your model here
        summary = self.summarizer(model_kwargs.field1, max_length=230, min_length=30, do_sample=False)


        return summary


with Pipeline() as builder:

    model_kwargs = Variable(ModelKwargs)

    model = Summarizer()
    model.load()

    summary = model.predict(model_kwargs)

    builder.output(summary)

my_new_pipeline = builder.get_pipeline()
