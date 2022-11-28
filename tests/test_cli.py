import pytest

# from pipeline.console import main as cli_main


@pytest.mark.parametrize("option", ("-h", "--help"))
def test_help():
    assert 0
