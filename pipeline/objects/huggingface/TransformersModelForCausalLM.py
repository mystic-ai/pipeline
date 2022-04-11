from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline import pipeline_function, pipeline_model


@pipeline_model()
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
    def predict(self, input_data: str, model_kwargs: dict) -> str:

        input_ids = self.tokenizer(input_data, return_tensors="pt").input_ids
        gen_tokens = self.model.generate(input_ids, **model_kwargs)
        gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
        return gen_text

    @pipeline_function
    def load(self) -> None:
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
