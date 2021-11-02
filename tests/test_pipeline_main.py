from pipeline import Pipeline


def test_check_running():

    with Pipeline("test") as my_pipeline:
        assert Pipeline._current_pipeline != None
        assert Pipeline._current_pipeline.name == "test"
