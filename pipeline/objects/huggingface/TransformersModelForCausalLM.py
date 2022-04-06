from pipeline import pipeline_model, pipeline_function
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Union, List


@pipeline_model()
class TransformersModelForCausalLM:
    def __init__(
        self,
        model_path: str,
        model_kwargs: dict,
        tokenizer_path: str,
    ):
        self.model_path = model_path
        self.model_kwargs = model_kwargs
        self.tokenizer_path = tokenizer_path
        self.model = None
        self.tokenizer = None

    @pipeline_function
    def predict(
        self,
        input: Union[str, List[str], List[int]],
        inference_kwargs: dict,
    ) -> str:
        assert any(
            [isinstance(input, str), isinstance(input, list)]
        ), "Input should be a string, a list of strings, or a list of integers."
        if isinstance(input, str):
            input = [input]
        input_ids = self.tokenizer.encode(input, return_tensors="pt")
        gen_tokens = self.model.generate(input_ids, **inference_kwargs)
        gen_text = self.tokenizer.batch_decode(gen_tokens)
        return gen_text

    @pipeline_function
    def load(self) -> None:
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, **self.model_kwargs
            )
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
