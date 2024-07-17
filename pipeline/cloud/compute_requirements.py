from enum import Enum


class Accelerator(str, Enum):
    nvidia_t4 = "nvidia_t4"
    nvidia_a100 = "nvidia_a100"
    nvidia_a100_80gb = "nvidia_a100_80gb"
    nvidia_h100 = "nvidia_h100"
    nvidia_l4 = "nvidia_l4"

    cpu = "cpu"

    nvidia_a100_5gb = "nvidia_a100_5gb"
    nvidia_a100_10gb = "nvidia_a100_10gb"
    nvidia_a100_20gb = "nvidia_a100_20gb"

    nvidia_a100_80gb_10gb = "nvidia_a100_80gb_10gb"
    nvidia_a100_80gb_20gb = "nvidia_a100_80gb_20gb"
    nvidia_a100_80gb_40gb = "nvidia_a100_80gb_40gb"

    nvidia_a10 = "nvidia_a10"
    nvidia_a10_12gb = "nvidia_a10_12gb"
    nvidia_a10_8gb = "nvidia_a10_8gb"
    nvidia_a10_4gb = "nvidia_a10_4gb"

    nvidia_h100_10gb = "nvidia_h100_10gb"
    nvidia_h100_20gb = "nvidia_h100_20gb"
    nvidia_h100_40gb = "nvidia_h100_40gb"
    nvidia_h100_80gb = "nvidia_h100_80gb"
