from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers.models.m2m_100.tokenization_m2m_100 import FAIRSEQ_LANGUAGE_CODES

from pipeline import Pipeline, Variable, entity, pipe
from pipeline.objects.graph import InputField, InputSchema


class ModelKwargs(InputSchema):
    source_language: str | None = InputField(
        default="en",
        optional=True,
        title="Source language",
        description="The language of the input text",
        choices=FAIRSEQ_LANGUAGE_CODES.get("m2m100", {}),
    )

    target_language: str | None = InputField(
        default="fr",
        optional=True,
        title="Target language",
        description="The language to translate to",
        choices=FAIRSEQ_LANGUAGE_CODES.get("m2m100", {}),
    )


@entity
class MyModelClass:
    @pipe(run_once=True, on_startup=True)
    def load(self) -> None:
        self.model = M2M100ForConditionalGeneration.from_pretrained(
            "facebook/m2m100_418M"
        )

        self.model = self.model.to("cuda")

        self.tokenizer = M2M100Tokenizer.from_pretrained(
            "facebook/m2m100_418M",
        )

    @pipe
    def predict(self, text: str, args: ModelKwargs) -> str:
        self.tokenizer.src_lang = args.source_language

        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = inputs.to("cuda")
        generated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.get_lang_id(args.target_language),
        )
        output_str = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )

        return output_str


with Pipeline() as builder:
    input_var = Variable(
        str,
        description="Input text to translate",
        title="Input text",
    )

    args = Variable(
        ModelKwargs,
        description="args for the model",
        title="Model args",
    )

    my_model = MyModelClass()
    my_model.load()

    output_var = my_model.predict(input_var, args)

    builder.output(output_var)

my_new_pipeline = builder.get_pipeline()
