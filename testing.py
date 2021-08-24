import npu2
from npu2 import pipeline
from npu2.object import Object, Scalar, Tensor
from npu2.pipeline import create_pipeline


@npu2.function.function_inputs(input_1=Scalar, input_2=Scalar)
@npu2.function.function_outputs(input_1=Scalar)
def pipeline_func1(input_1, input_2):
    return {"input_1": input_1 + input_2}


@npu2.function.function_inputs(input_1=Scalar)
def pipeline_func2(input_1):
    print(input_1)


pipeline_array = [pipeline_func1, pipeline_func2]

new_pipeline = create_pipeline(pipeline_array)


print(new_pipeline.run(input_1=5, input_2=8))
