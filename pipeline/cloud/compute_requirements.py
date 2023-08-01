from enum import Enum


class Accelerator(str, Enum):
    nvidia_t4: str = "nvidia_t4"
    nvidia_a100: str = "nvidia_a100"
    nvidia_a100_80gb: str = "nvidia_a100_80gb"
    nvidia_v100: str = "nvidia_v100"
    nvidia_v100_32gb: str = "nvidia_v100_32gb"
    nvidia_3090: str = "nvidia_3090"
    nvidia_a16: str = "nvidia_a16"
    nvidia_h100: str = "nvidia_h100"
    nvidia_all: str = "nvidia_all"
    cpu: str = "cpu"


nvidia_gpus = [
    Accelerator.nvidia_t4,
    Accelerator.nvidia_a100,
    Accelerator.nvidia_a100_80gb,
    Accelerator.nvidia_v100,
    Accelerator.nvidia_v100_32gb,
    Accelerator.nvidia_3090,
    Accelerator.nvidia_a16,
    Accelerator.nvidia_h100,
    Accelerator.nvidia_all,
]
