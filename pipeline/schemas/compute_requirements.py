from typing import Optional

from .base import BaseModel


class ComputeRequirements(BaseModel):
    """Additional compute requirements required by a runnable (e.g. min GPU VRAM)"""

    min_gpu_vram_mb: Optional[int]
