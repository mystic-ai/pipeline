import pytest


def test_imports():
    """All submodules should be import'able."""
    try:
        from pipeline.schemas import (
            base,
            data,
            file,
            function,
            resource,
            run,
            runnable,
            tag,
            token,
        )
    except ImportError:
        pytest.fail("unable to import modules")
