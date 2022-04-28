"""Request/response schemas for user onboarding tasks."""
from typing import Optional

from pydantic import HttpUrl

from .base import BaseModel


class OnboardingTaskGet(BaseModel):
    """Response schema describing a single onboarding task."""

    #: Database identifier
    id: str
    #: Shorthand name; unique across tasks
    name: str
    #: Longform description
    description: Optional[str]
    #: Approximate time required to complete this task
    time_to_complete_min: int
    #: URL to further information
    url: Optional[HttpUrl]
    #: Completion status
    complete: bool


class OnboardingTaskPatch(BaseModel):
    """Patch schema for updating a single onboarding task."""

    #: Completion status
    complete: bool
