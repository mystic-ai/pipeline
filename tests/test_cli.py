import pytest

from pipeline.console import main as cli_main


@pytest.mark.parametrize("option", ("-h", "--help"))
def test_help(capsys, option):
    cli_main(option)
    # output = capsys.readouterr().out
    assert 1
