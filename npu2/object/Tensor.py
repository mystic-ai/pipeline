import numpy as np

from npu2.object import Object
class Tensor(Object):
    name: str
    value: np.ndarray