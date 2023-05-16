import datetime

from pipeline import Pipeline, Variable, pipeline_function
from pipeline.v3 import upload_pipeline


@pipeline_function
def pi_sample(i: int) -> float:
    import numpy as np

    x, y = np.random.rand(2)
    return bool(x**2 + y**2 < 1.0)


with Pipeline("pi-approx") as builder:
    input_var = Variable(int, is_input=True)
    builder.add_variables(input_var)
    b = pi_sample(input_var)
    builder.output(b)

pl = Pipeline.get_pipeline("pi-approx")

start_time = datetime.datetime.now()

result = upload_pipeline(pl)

end_time = datetime.datetime.now()

total_time = (end_time - start_time).total_seconds()

print(f"Total time taken: {total_time}, result: {result.content}")
