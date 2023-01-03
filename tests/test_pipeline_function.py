from typing import Tuple

import pytest

from pipeline import Pipeline, pipeline_function


def test_function_no_output_definition():
    @pipeline_function
    def test_function():  # No output defined
        ...

    with Pipeline("test"):
        test_function()

    test_pl = Pipeline.get_pipeline("test")

    assert len(test_pl.functions) == 1
    assert len(test_pl.nodes) == 1


def test_function_tuple_output():
    @pipeline_function
    def test_function() -> Tuple[str, int]:
        return ("test", 1)

    with Pipeline("test") as builder:
        var1, var2 = test_function()

        builder.output(var2, var1)

    test_pl = Pipeline.get_pipeline("test")

    outputs = test_pl.run()

    assert outputs == [1, "test"]

    @pipeline_function
    def test_function_2(input_1: str):
        ...

    with pytest.raises(Exception):
        with Pipeline("test2") as builder:
            var = test_function()
            test_function_2(var)
