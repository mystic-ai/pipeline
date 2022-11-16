import torch
import copy

import numpy as np
from typing import List, Tuple, Dict

from pipeline.objects import pipeline_function


@pipeline_function
def tensor_to_list(input_tensor: torch.Tensor) -> list:
    return np.array(input_tensor.cpu().detach().numpy(), dtype=np.float64).tolist()


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def extract_tensors(m: torch.nn.Module) -> Tuple[torch.nn.Module, List[Dict]]:
    tensors = []
    for _, module in m.named_modules():
        params = {
            name: torch.clone(param).detach().numpy()
            for name, param in module.named_parameters(recurse=False)
        }
        buffers = {
            name: torch.clone(buf).detach().numpy()
            for name, buf in module.named_buffers(recurse=False)
        }
        tensors.append({"params": params, "buffers": buffers})
    m_copy = copy.deepcopy(m)
    for _, module in m_copy.named_modules():
        for name in [name for name, _ in module.named_parameters(recurse=False)] + [
            name for name, _ in module.named_buffers(recurse=False)
        ]:
            setattr(module, name, None)

    m_copy.train(False)
    return m_copy, tensors


def replace_tensors(m: torch.nn.Module, tensors: List[Dict]) -> None:
    modules = [module for _, module in m.named_modules()]
    for module, tensor_dict in zip(modules, tensors):
        for name, array in tensor_dict["params"].items():
            module.register_parameter(name, torch.nn.Parameter(torch.as_tensor(array)))
        for name, array in tensor_dict["buffers"].items():
            module.register_buffer(name, torch.as_tensor(array))
