from pipeline import Pipeline


# Check if the decorator correctly uses __init__ and __enter__
def test_with_decorator():
    with Pipeline("test") as my_pipeline:
        assert Pipeline._current_pipeline != None


# Check naming
def test_with_decorator_name():
    with Pipeline("test") as my_pipeline:
        assert Pipeline._current_pipeline.name == "test"


# Test exit
def test_with_exit():
    with Pipeline("test") as my_pipeline:
        ...
    assert Pipeline.get_pipeline("test").name == "test"
