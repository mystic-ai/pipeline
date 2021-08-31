import npu2
from npu2 import pipeline
from npu2.object import Object, Scalar, Tensor
from npu2.pipeline import create_pipeline, save_pipeline, load_pipeline
from npu2.function import save_function, load_function


@npu2.function(inputs={"input_1":Scalar, "input_2":Scalar}, outputs={"input_1":Scalar})
def pipeline_func1(input_1, input_2):
    return ((), {"input_1": input_1 + input_2})


@npu2.function(inputs={"input_1":Scalar})
def pipeline_func2(input_1):
    return (input_1, {})

#save_function(pipeline_func1, "./func1")
#new_func = load_function("./func1")
#print(new_func)

pipeline_array = [pipeline_func1, pipeline_func2]

new_pipeline = create_pipeline(pipeline_array)

#print(new_pipeline.run(input_1=5, input_2=8))
save_pipeline(new_pipeline, "./first_pipeline/")
new_pipeline = load_pipeline("./first_pipeline/")
print(new_pipeline.run(90,48))

npu2.link("0197246120897461209476120983613409861230986")

npu2.upload(pipeline_func1)