import npu2
from npu2 import pipeline
from npu2.object import Object, Scalar, Tensor
from npu2.pipeline import create_pipeline, save_pipeline, load_pipeline
from npu2.function import save_function, load_function


@npu2.function(
    inputs={"input_1": Object}, outputs={"input_1": Object}
)
def func1(input_1: str):
    return ((), {"input_1": input_1 + " lol"})



import npu2.api

npu2.api.API_ENDPOINT = "http://localhost:5002/v2"
npu2.link("0197246120897461209476120983613409861230986")

# print(npu2.upload(pipeline_func1))
print(npu2.upload(func1))
