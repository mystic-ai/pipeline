from enum import Enum
from typing import Optional

from .base import BaseModel


# https://github.com/samuelcolvin/pydantic/issues/2278
class ComputeType(str, Enum):
    cpu: str = "cpu"
    gpu: str = "gpu"


class ComputeRequirements(BaseModel):
    """Additional compute requirements required by a runnable (e.g. min GPU VRAM)"""

    min_gpu_vram_mb: Optional[int]
