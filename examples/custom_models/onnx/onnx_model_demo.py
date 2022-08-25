from pipeline import (
    onnx_model,
    PipelineCloud
)
from pipeline.util.torch_utils import tensor_to_list
import torch

onnx_pipeline = onnx_model('./example.onnx')

#run locally
input = tensor_to_list(torch.rand(1, 28, 28, device='cpu'))
# result = onnx_pipeline.run(['output'],{'input':input})
# api = PipelineCloud(token="neuro_sk_aLGaopgrjCuomO6PV2BKFVNFIuR2sAM5")
api = PipelineCloud(url="http://10.0.0.151:5000", token="sudo1234")
print("Now uploading")
uploaded_pipeline = api.upload_pipeline(onnx_pipeline)

run_result = api.run_pipeline(
    uploaded_pipeline, [['output'],{'input':input}]
)
breakpoint()
try:
    result_preview = run_result["result_preview"]
except KeyError:
    result_preview = "unavailable"
print("Run result:", result_preview)
