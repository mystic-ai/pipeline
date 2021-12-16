from pipeline.objects import Pipeline, pipeline_function


# Check if the decorator correctly uses __init__ and __enter__
def test_with_decorator():
    with Pipeline("test"):
        assert Pipeline._current_pipeline is not None


# Check naming
def test_with_decorator_name():
    with Pipeline("test"):
        assert Pipeline._current_pipeline.name == "test"


# Test exit
def test_exit():
    with Pipeline("test") as pipeline:
        pipeline.add_variable(type_class=float, is_input=True, is_output=True)
    assert Pipeline.get_pipeline("test").name == "test"


# Test basic Pipeline
def test_basic_pipeline():
    @pipeline_function
    def add(f_1: float, f_2: float) -> float:
        return f_1 + f_2

    @pipeline_function
    def square(f_1: float) -> float:
        return f_1 ** 2

    assert Pipeline._current_pipeline is None
    with Pipeline("basic_test") as my_pipeline:
        in_1 = my_pipeline.add_variable(float, is_input=True)
        in_2 = my_pipeline.add_variable(float, is_input=True)

        add_1 = add(in_1, in_2)
        sq_1 = square(add_1)

        my_pipeline.output(sq_1, add_1)

    output = Pipeline.run("basic_test", 2.0, 3.0)
    assert output == [25.0, 5.0]
    assert Pipeline._current_pipeline is None
