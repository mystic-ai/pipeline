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

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


@pipeline_model
class SentenceEmbedding:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    @pipeline_function
    def predict(self, input_data: str) -> list:
        # Tokenize sentences
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return tensor_to_list(sentence_embeddings)

    @pipeline_function(run_once=True, on_startup=True)
    def load(self) -> None:
        from transformers import AutoTokenizer, BertModel
        
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model.to(0)


api = PipelineCloud()

with Pipeline("all-MiniLM-L6-v2") as builder:
    input_str = Variable(str, is_input=True)

    builder.add_variables(
        input_str,
    )

    model = SentenceEmbedding()
    model.load()
    output_str = model.predict(
        input_str,
    )

    builder.output(output_str)

output_pipeline = Pipeline.get_pipeline("all-MiniLM-L6-v2")
uploaded_pipeline = api.upload_pipeline(output_pipeline)
print(uploaded_pipeline)
