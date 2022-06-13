from typing import Optional

from .base import BaseModel


class MilestonesRegisterGet(BaseModel):
    """View of a users' milestones register"""

    #: Has the user ever deployed a pipeline?
    pipeline_deployed: bool
    #: Has the user ever executed a pipeline run?
    run_executed: bool
    #: Has the user ever visited dashboard modelhub?
    modelhub_viewed: bool


class MilestonesRegisterPatch(BaseModel):
    """Patch a users' milestones register"""

    # `pipeline_deployed` & `run_executed` update managed internally

    modelhub_viewed: Optional[bool]
