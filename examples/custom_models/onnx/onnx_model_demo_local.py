import torch

from pipeline import onnx_to_pipeline
from pipeline.util.torch_utils import tensor_to_list

onnx_pipeline = onnx_to_pipeline("./example.onnx")

# run locally
input = tensor_to_list(torch.rand(1, 28, 28, device="cpu"))
result = onnx_pipeline.run(["output"], {"input": input})[0]

print(result)
