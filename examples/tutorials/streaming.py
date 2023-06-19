import time

from pipeline import Pipeline, pipeline_function
from pipeline.objects.variable import Stream, Variable


@pipeline_function
def streaming_function(input_str: str) -> Stream[str]:
    for i in range(10):
        time.sleep(0.5)
        yield str(i)


with Pipeline("streaming-test") as builder:
    input_str = Variable(type_class=str, is_input=True)
    builder.add_variables(input_str)
    output_str = streaming_function(input_str)

    builder.output(output_str)

pl = Pipeline.get_pipeline("streaming-test")

iterator = pl.run("Test meeeeee")

for i in iterator[0]:
    print(i)
