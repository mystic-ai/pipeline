from pipeline import pipeline_model, pipeline_function
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Union, List


@pipeline_model()
class TransformersModelForCausalLM:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        model_kwargs: dict = {},
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model_kwargs = model_kwargs
        self.model = None
        self.tokenizer = None

    @pipeline_function
    def predict(
        self,
        inputs: Union[str, List[str], List[int]],
        inference_kwargs: dict,
    ) -> str:
        assert any(
            [isinstance(inputs, str), isinstance(inputs, list)]
        ), "Input should be a string, a list of strings, or a list of integers."
        if isinstance(inputs, str):
            inputs = [inputs]
        inputs_ids = self.tokenizer.encode(inputs, return_tensors="pt")
        gen_tokens = self.model.generate(inputs_ids, **inference_kwargs)
        output = self.tokenizer.batch_decode(gen_tokens)
        return output

    @pipeline_function
    def load(self) -> None:
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, **self.model_kwargs
            )
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
