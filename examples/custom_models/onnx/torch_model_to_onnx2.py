import torch
from torch import nn
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x,y):
        y = self.flatten(y)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x) +  self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().cpu()

dummy_input = torch.rand(1, 28, 28, device='cpu')
logits = model(dummy_input, dummy_input)


input_names = [ "input1","input2" ]
output_names = [ "output" ]
dummy_input2 = torch.rand(1, 28, 28, device='cpu')
torch.onnx.export(model, (dummy_input, dummy_input2), "example2.onnx", verbose=True, input_names=input_names, output_names=output_names)
import onnxruntime
session = onnxruntime.InferenceSession(
                "example2.onnx",
                providers=[
                    "CUDAExecutionProvider",
                ],)
input = torch.rand(1, 28, 28, device='cpu')
session.run(output_names, {"input1":input,"input2":input})