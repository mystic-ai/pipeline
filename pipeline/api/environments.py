"""Pre-made public environments your pipelines can execute in."""
from collections import namedtuple
from typing import Optional, Union

PipelineCloudEnvironment = namedtuple("PipelineCloudEnvironment", ["id", "name"])

# TODO: add instructions for viewing this environment's contents.
mystic_default_20230126 = PipelineCloudEnvironment(
    id="environment_35a54969b7474150b87afdf155431884",
    name="public/mystic-default-20230126",
)

DEFAULT_ENVIRONMENT = mystic_default_20230126


def resolve_environment_id(
    environment: Optional[Union[PipelineCloudEnvironment, str]] = None
):
    """Resolve the argument to a remote Environment ID.

    Fall back to a 'default' environment hard-coded on the client.
    """
    if environment is None:
        environment = DEFAULT_ENVIRONMENT
    try:
        environment_id = environment.id
    except AttributeError:
        # Assume `environment` is a str ID.
        environment_id = environment
    return environment_id
