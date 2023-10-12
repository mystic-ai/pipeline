from collections import defaultdict
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
    nvidia_l4: str = "nvidia_l4"
    nvidia_a5000: str = "nvidia_a5000"
    nvidia_all: str = "nvidia_all"
    cpu: str = "cpu"

    @classmethod
    def from_str(cls, accelerator: str) -> "Accelerator":
        if "T4" in accelerator:
            accelerator_type = Accelerator.nvidia_t4
        elif "A100" in accelerator:
            if "80GB" in accelerator:
                accelerator_type = Accelerator.nvidia_a100_80gb
            else:
                accelerator_type = Accelerator.nvidia_a100
        elif "H100" in accelerator:
            accelerator_type = Accelerator.nvidia_h100
        elif "V100" in accelerator:
            if "32GB" in accelerator:
                accelerator_type = Accelerator.nvidia_v100_32gb
            else:
                accelerator_type = Accelerator.nvidia_v100
        elif "3090" in accelerator:
            accelerator_type = Accelerator.nvidia_3090
        elif "A16" in accelerator:
            accelerator_type = Accelerator.nvidia_a16
        elif "L4" in accelerator:
            accelerator_type = Accelerator.nvidia_l4
        elif "A5000" in accelerator:
            accelerator_type = Accelerator.nvidia_a5000
        else:
            raise Exception(f"Unknown GPU name: {accelerator}")

        return accelerator_type

    @classmethod
    def valid_accelerator_config(cls, accelerators: list["Accelerator"]):
        return accelerators in [
            [Accelerator.nvidia_t4],
            [Accelerator.nvidia_a100],
            [Accelerator.nvidia_a100] * 2,
            [Accelerator.nvidia_a100] * 4,
            [Accelerator.nvidia_a100_80gb],
            [Accelerator.nvidia_a100_80gb] * 2,
            [Accelerator.nvidia_a100_80gb] * 4,
            [Accelerator.nvidia_v100],
            [Accelerator.nvidia_v100_32gb],
            [Accelerator.nvidia_l4],
            [Accelerator.nvidia_a5000],
            [Accelerator.nvidia_all],
            [Accelerator.cpu],
        ]

    def max_memory_mb(self) -> int:
        return defaultdict(
            lambda: 10_000,
            {
                Accelerator.nvidia_t4: 16_000,
                Accelerator.nvidia_a100: 40_000,
                Accelerator.nvidia_a100_80gb: 80_000,
                Accelerator.nvidia_v100: 16_000,
                Accelerator.nvidia_v100_32gb: 32_000,
                Accelerator.nvidia_l4: 22_000,
                Accelerator.nvidia_a5000: 24_000,
            },
        )[self]


nvidia_gpus = [
    Accelerator.nvidia_t4,
    Accelerator.nvidia_a100,
    Accelerator.nvidia_a100_80gb,
    Accelerator.nvidia_v100,
    Accelerator.nvidia_v100_32gb,
    Accelerator.nvidia_3090,
    Accelerator.nvidia_a16,
    Accelerator.nvidia_h100,
    Accelerator.nvidia_l4,
    Accelerator.nvidia_a5000,
    Accelerator.nvidia_all,
]
