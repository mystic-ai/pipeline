from enum import Enum


class Accelerator(str, Enum):
    nvidia_t4: str = "nvidia_t4"
    nvidia_a100: str = "nvidia_a100"
    nvidia_a100_80gb: str = "nvidia_a100_80gb"
    nvidia_h100: str = "nvidia_h100"
    nvidia_l4: str = "nvidia_l4"

    cpu: str = "cpu"

    nvidia_a100_5gb: str = "nvidia_a100_5gb"
    nvidia_a100_10gb: str = "nvidia_a100_10gb"
    nvidia_a100_20gb: str = "nvidia_a100_20gb"

    nvidia_a100_80gb_10gb: str = "nvidia_a100_80gb_10gb"
    nvidia_a100_80gb_20gb: str = "nvidia_a100_80gb_20gb"
    nvidia_a100_80gb_40gb: str = "nvidia_a100_80gb_40gb"

    nvidia_a10: str = "nvidia_a10"

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
        elif "L4" in accelerator:
            accelerator_type = Accelerator.nvidia_l4
        # guess this works as long as its after A100
        elif "A10" in accelerator:
            accelerator_type = Accelerator.nvidia_a10

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
            [Accelerator.nvidia_l4],
            [Accelerator.nvidia_a10],
            [Accelerator.cpu],
        ]


nvidia_gpus = [
    Accelerator.nvidia_t4,
    Accelerator.nvidia_a100,
    Accelerator.nvidia_a100_80gb,
    Accelerator.nvidia_h100,
    Accelerator.nvidia_l4,
    Accelerator.nvidia_a100_5gb,
    Accelerator.nvidia_a100_10gb,
    Accelerator.nvidia_a100_20gb,
    Accelerator.nvidia_a100_80gb_10gb,
    Accelerator.nvidia_a100_80gb_20gb,
    Accelerator.nvidia_a100_80gb_40gb,
    Accelerator.nvidia_a10,
]
