from pipeline import pipeline_function
from pipeline.model import pipeline_model

from transformers import AutoModelForCausalLM, AutoTokenizer


@pipeline_model()
class TransformersModelForCausalLM(object):
    model: AutoModelForCausalLM = None
    tokenizer: AutoTokenizer = None

    def __init__(self, model_path="", tokenizer_path=""):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

    @pipeline_function
    def predict(self, input: str, **kwargs: dict) -> str:
        input_ids = self.tokenizer(input, return_tensors="pt").input_ids
        output_tokens = self.model.generate(input_ids, **kwargs)
        output_str = self.tokenizer.batch_decode(output_tokens)[0]
        return output_str

    @pipeline_function
    def load(self, path: str) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
