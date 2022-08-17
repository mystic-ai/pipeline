from pipeline.objects import Pipeline, Variable, pipeline_function
from ..pipeline.objects.decorators import pipeline_model


# Check if the decorator correctly uses __init__ and __enter__
def test_with_decorator():
    with Pipeline("test"):
        assert Pipeline._current_pipeline is not None


# Check naming
def test_with_decorator_name():
    with Pipeline("test"):
        assert Pipeline._current_pipeline.name == "test"


# Test exit
def test_with_exit():
    with Pipeline("test"):
        Variable(str, is_input=True, is_output=True)
    assert Pipeline.get_pipeline("test").name == "test"


# Test basic Pipeline
def test_basic_pipeline():
    @pipeline_function
    def add(f_1: float, f_2: float) -> float:
        return f_1 + f_2

    @pipeline_function
    def square(f_1: float) -> float:
        return f_1**2

    with Pipeline("test") as my_pipeline:
        in_1 = Variable(float, is_input=True)
        in_2 = Variable(float, is_input=True)

        my_pipeline.add_variables(in_1, in_2)

        add_1 = add(in_1, in_2)
        sq_1 = square(add_1)

        my_pipeline.output(sq_1, add_1)

    output_pipeline = Pipeline.get_pipeline("test")

    output = output_pipeline.run(2.0, 3.0)
    assert output == [25.0, 5.0]
    assert Pipeline._current_pipeline is None


def test_pipeline_with_compute_requirements(pipeline_graph_with_compute_requirements):
    pipeline_graph = pipeline_graph_with_compute_requirements
    assert pipeline_graph.compute_type == "gpu"
    assert pipeline_graph.min_gpu_vram_mb == 4000


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

    with Pipeline("test") as builder:
        # in_1 = Variable(int, is_input=True)
        # builder.add_variable(in_1)
        my_simple_model = simple_model()
        my_simple_model.run_once_func()
        my_simple_model.run_once_func()
        output = my_simple_model.get_number()
        builder.output(output)

    output_pipeline = Pipeline.get_pipeline("test")
    output_number = output_pipeline.run()
    assert output_number == 1
