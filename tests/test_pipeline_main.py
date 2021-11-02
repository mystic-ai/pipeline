from pipeline import Pipeline


def test_with_decorator():

    with Pipeline("test") as my_pipeline:
        assert Pipeline._current_pipeline != None


def test_with_decorator_name():

    with Pipeline("test") as my_pipeline:
        assert Pipeline._current_pipeline.name == "test"
