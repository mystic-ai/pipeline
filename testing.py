import npu2
from npu2.object import Object, Scalar, Tensor


@npu2.function.function_inputs(input_1=Scalar, input_2=Scalar)
@npu2.function.function_outputs(output_1=Scalar)
def my_static_func(input_1, input_2):
    return {"output_1" : input_1 + input_2}


print(dir(my_static_func))
print(my_static_func.__npu_outputs__)