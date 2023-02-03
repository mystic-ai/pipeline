from pipeline.api.environments import DEFAULT_ENVIRONMENT, resolve_environment_id


def test_environment_resolution_default():
    """A `None` identifier should resolve to the default environment's ID."""
    assert resolve_environment_id(environment=None) == DEFAULT_ENVIRONMENT.id


def test_environment_resolution_string():
    """A string indentifier should resolve to that string."""
    value = "environment_abc123"
    assert resolve_environment_id(environment=value) == value


def test_environment_resolution_object():
    """An object with an `id` property should resolve to the value of that property."""

    class IDHolder:
        id: str = "environment_cab321"

    obj = IDHolder()
    assert resolve_environment_id(environment=obj) == obj.id
