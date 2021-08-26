import npu2
from npu2 import pipeline
from npu2.object import Object, Scalar, Tensor
from npu2.pipeline import create_pipeline, save_pipeline, load_pipeline


@npu2.function(inputs={"input_1":Scalar, "input_2":Scalar}, outputs={"input_1":Scalar})
def pipeline_func1(input_1, input_2):
    return ((), {"input_1": input_1 + input_2})


@npu2.function(inputs={"input_1":Scalar})
def pipeline_func2(input_1):
    return (input_1, {})


pipeline_array = [pipeline_func1, pipeline_func2]

new_pipeline = create_pipeline(pipeline_array)

print(new_pipeline.run(input_1=5, input_2=8))
save_pipeline(new_pipeline, "./first_pipeline/")
new_pipeline = load_pipeline("./first_pipeline/")
print(new_pipeline.run(90,48))

