from typing import Tuple

import pytest

from pipeline import Pipeline, Variable, pipe


def test_function_no_output_definition():
    @pipe
    def test_function():  # No output defined
        ...

    with Pipeline() as builder:
        test_function()

    test_pl = builder.get_pipeline()

    assert len(test_pl.functions) == 1
    assert len(test_pl.nodes) == 1


def test_basic_function():
    @pipe
    def return_inverse(in_bool: bool) -> bool:
        return not in_bool

    with Pipeline() as builder:
        in_bool = Variable(bool)
        output_bool = return_inverse(in_bool)
        builder.output(output_bool)

    test_pl = builder.get_pipeline()

    assert not test_pl.run(True)[0]


def test_function_tuple_output():
    @pipe
    def test_function() -> Tuple[str, int]:
        return ("test", 1)

    with Pipeline() as builder:
        var1, var2 = test_function()

        builder.output(var2, var1)

    test_pl = builder.get_pipeline()

    outputs = test_pl.run()

    assert outputs == [1, "test"]

    @pipe
    def test_function_2(input_1: str):
        ...

    with pytest.raises(Exception):
        with Pipeline() as builder:
            var = test_function()
            test_function_2(var)
