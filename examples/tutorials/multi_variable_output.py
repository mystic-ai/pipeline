from typing import Tuple

from pipeline import Pipeline, pipeline_function


@pipeline_function
def test_function() -> Tuple[str, int]:
    return ("test", 1)


with Pipeline("test") as builder:
    var1, var2 = test_function()  # Must split variables, cannot output raw Tuple
    builder.output(var2, var1)


test_pl = Pipeline.get_pipeline("test")
outputs = test_pl.run()

assert outputs == [1, "test"]
