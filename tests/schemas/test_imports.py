def test_imports():
    """All submodules should be import'able."""
    from pipeline.schemas import (
        base,
        data,
        file,
        function,
        project,
        resource,
        run,
        runnable,
        tag,
        token,
    )
