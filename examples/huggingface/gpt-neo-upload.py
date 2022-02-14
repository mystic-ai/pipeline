"""from pipeline.objects.huggingface.TransformersModelForCausalLM import (
    TransformersModelForCausalLM,
)"""


from pipeline import Pipeline, PipelineCloud, Variable

from pipeline import pipeline_model, pipeline_function
from transformers import AutoModelForCausalLM, AutoTokenizer


@pipeline_model
class TransformersModelForCausalLM:
    def __init__(
        self,
        model_path: str = "EleutherAI/gpt-neo-125M",
        tokenizer_path: str = "EleutherAI/gpt-neo-125M",
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model = None
        self.tokenizer = None

    @pipeline_function
    def predict(self, input_data: str, model_kwargs: dict = {}) -> str:
        input_ids = self.tokenizer(input_data, return_tensors="pt").input_ids
        gen_tokens = self.model.generate(input_ids, **model_kwargs)
        gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
        return gen_text

    @pipeline_function
    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import transformers

        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)


# load_dotenv("hidden.env")

api = PipelineCloud()
api.authenticate()

with Pipeline("HF pipeline") as builder:
    input_str = Variable(str, is_input=True)
    model_kwargs = Variable(dict, is_input=True)

    builder.add_variables(
        input_str,
        model_kwargs,
    )

    hf_model = TransformersModelForCausalLM(
        model_path="EleutherAI/gpt-neo-125M",
        tokenizer_path="EleutherAI/gpt-neo-125M",
    )

    output_str = hf_model.predict(
        input_str,
        model_kwargs,
    )

    builder.output(output_str)

output_pipeline = Pipeline.get_pipeline("HF pipeline")

# print(output_pipeline.run("Hello there"))

print("Now uploading GPTNeo pipeline")
uploaded_pipeline = api.upload_pipeline(output_pipeline)
print(uploaded_pipeline)
print(
    api.run_pipeline(
        uploaded_pipeline,
        ["Hello my name is", {"max_length": 100}],
    )
)
