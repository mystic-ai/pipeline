# from transformers import AutoTokenizer, BertModel
# import torch

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertModel.from_pretrained("bert-base-uncased")

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_state

from pipeline import (
    Pipeline,
    PipelineCloud,
    Variable,
    pipeline_function,
    pipeline_model,
)
from pipeline.util.torch_utils import tensor_to_list


@pipeline_model
class BertEmbedding:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    @pipeline_function
    def predict(self, input_data: str) -> list:
        inputs = self.tokenizer(input_data, return_tensors="pt").to(0)
        outputs = self.model(**inputs)
        return tensor_to_list(outputs.last_hidden_state)

    @pipeline_function(run_once=True, on_startup=True)
    def load(self) -> None:
        from transformers import AutoTokenizer, BertModel

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model.to(0)


api = PipelineCloud()

with Pipeline("bert") as builder:
    input_str = Variable(str, is_input=True)

    builder.add_variables(
        input_str,
    )

    model = BertEmbedding()
    model.load()
    output_str = model.predict(
        input_str,
    )

    builder.output(output_str)

output_pipeline = Pipeline.get_pipeline("bert")
uploaded_pipeline = api.upload_pipeline(output_pipeline)
print(uploaded_pipeline)
