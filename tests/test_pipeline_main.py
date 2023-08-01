import pytest

from pipeline.objects import (
    Pipeline,
    PipelineFile,
    Variable,
    pipeline_function,
    pipeline_model,
)


# Check if the decorator correctly uses __init__ and __enter__
def test_with_decorator():
    with Pipeline() as builder:
        assert builder._current_pipeline is not None


# Test basic Pipeline
def test_basic_pipeline():
    @pipeline_function
    def add(f_1: float, f_2: float) -> float:
        return f_1 + f_2

    @pipeline_function
    def square(f_1: float) -> float:
        return f_1**2

    with Pipeline() as builder:
        in_1 = Variable(float)
        in_2 = Variable(float)

        add_1 = add(in_1, in_2)
        sq_1 = square(add_1)

        builder.output(sq_1, add_1)

    output_pipeline = builder.get_pipeline()

    output = output_pipeline.run(2.0, 3.0)
    assert output == [25.0, 5.0]


def test_run_once():
    @pipeline_model
    class simple_model:
        def __init__(self):
            self.test_number = 0

        @pipeline_function(run_once=True)
        def run_once_func(self) -> int:
            self.test_number += 1
            return self.test_number

        @pipeline_function
        def get_number(self) -> int:
            return self.test_number

    with Pipeline() as builder:
        my_simple_model = simple_model()
        my_simple_model.run_once_func()
        my_simple_model.run_once_func()
        output = my_simple_model.get_number()
        builder.output(output)

    output_pipeline = builder.get_pipeline()
    output_number = output_pipeline.run()
    assert output_number == [1]


def test_run_startup():
    @pipeline_model
    class simple_model:
        def __init__(self):
            self.test_number = 0

        @pipeline_function(on_startup=True)
        def run_startup_func(self) -> int:
            self.test_number += 1
            return self.test_number

        @pipeline_function
        def get_number(self) -> int:
            return self.test_number

    with Pipeline() as builder:
        my_simple_model = simple_model()
        output = my_simple_model.get_number()
        # The run_startup_func is called after the get_number in the pipeline,
        # but as a startup func it will actually be called before.

        my_simple_model.run_startup_func()
        builder.output(output)

    output_pipeline = builder.get_pipeline()
    output_number = output_pipeline.run()
    assert output_number == [1]


def test_remote_file_not_downloaded():
    with Pipeline() as builder:
        PipelineFile()

    test_pipeline = builder.get_pipeline()
    with pytest.raises(
        Exception,
        match="Must call PipelineCloud()",
    ):
        test_pipeline.run()
