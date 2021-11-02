import numpy as np
from npu2.object import Object

class Scalar(Object):
    name: str
    value: np.float64