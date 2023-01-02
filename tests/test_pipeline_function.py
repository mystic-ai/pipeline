from typing import Tuple

from pipeline import Pipeline, pipeline_function


def test_function_no_output_definition():
    @pipeline_function
    def bad_func():  # No output defined
        ...

    with Pipeline("test"):
        bad_func()

    test_pl = Pipeline.get_pipeline("test")

    assert len(test_pl.functions) == 1
    assert len(test_pl.nodes) == 1


def test_function_tuple_output():
    @pipeline_function
    def bad_func() -> Tuple[str, int]:
        return ("test", 1)

    with Pipeline("test") as builder:
        var1, var2 = bad_func()

        builder.output(var2, var1)

    test_pl = Pipeline.get_pipeline("test")

    outputs = test_pl.run()

    assert outputs == [1, "test"]
