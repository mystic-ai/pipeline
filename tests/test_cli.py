import pytest

from pipeline.console import main as cli_main


@pytest.mark.parametrize("option", ("-h", "--help"))
def test_help(capsys, option):
    try:
        cli_main([option])
    except SystemExit:
        ...
    output: str = capsys.readouterr().out
    print(output, flush=True)
    assert output.startswith("usage: pipeline")
