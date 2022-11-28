import pytest

from pipeline.console import main as cli_main


@pytest.mark.parametrize("option", ("-h", "--help"))
def test_help(capsys, option):
    try:
        cli_main([option])
    except SystemExit:
        ...
    output: str = capsys.readouterr().out
    assert output.startswith("usage: pipeline")
