import numpy as np
import torch

from pipeline.objects import pipeline_function


@pipeline_function
def tensor_to_list(input_tensor: torch.Tensor) -> list:
    return np.array(input_tensor.cpu().detach().numpy(), dtype=np.float64).tolist()
