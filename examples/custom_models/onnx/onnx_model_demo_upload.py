import torch

from pipeline import PipelineCloud, onnx_to_pipeline
from pipeline.util.torch_utils import tensor_to_list

onnx_pipeline = onnx_to_pipeline("./example.onnx")

api = PipelineCloud(token="neuro_sk_aLGaopgrjCuomO6PV2BKFVNFIuR2sAM5")
uploaded_pipeline = api.upload_pipeline(onnx_pipeline)
print(f"Uploaded pipeline: {uploaded_pipeline.id}")

print("Run uploaded pipeline")
input = tensor_to_list(torch.rand(1, 28, 28, device="cpu"))
run_result = api.run_pipeline(uploaded_pipeline, [["output"], {"input": input}])
try:
    result_preview = run_result["result_preview"]
except KeyError:
    result_preview = "unavailable"
print("Run result:", result_preview)
