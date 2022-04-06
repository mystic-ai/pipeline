from pipeline import pipeline_model, pipeline_function
from transformers import AutoModelForMaskedLM, AutoTokenizer
from typing import Union, List


@pipeline_model()
class TransformersModelForMaskedLM:
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
        labels: Union[str, List[str], List[int]],
        inference_kwargs: dict,
    ) -> str:
        assert any(
            [isinstance(inputs, str), isinstance(inputs, list)]
        ), "Input should be a string, a list of strings, or a list of integers."
        if isinstance(inputs, str):
            inputs = [inputs]
        assert any(
            [isinstance(labels, str), isinstance(labels, list)]
        ), "Label should be a string, a list of strings, or a list of integers."
        if isinstance(labels, str):
            labels = [labels]
        inputs_ids = self.tokenizer.encode(inputs, return_tensors="pt")
        labels_ids = self.tokenizer.encode(labels, return_tensors="pt")
        output = self.model(inputs_ids, labels=labels_ids, **inference_kwargs)
        return output

    @pipeline_function
    def load(self) -> None:
        if self.model is None:
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.model_path, **self.model_kwargs
            )
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
