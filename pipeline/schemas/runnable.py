import enum
from typing import List, Optional

from pydantic import Field

from .base import BaseModel
from .project import ProjectGet


class RunnableType(enum.Enum):
    function = "function"
    pipeline = "pipeline"


class RunnableIOGet(BaseModel):
    """Unrealised/expected input/output data to a Runnable."""

    name: str
    type: str


class RunnableGet(BaseModel):
    id: str
    type: str
    name: Optional[str]
    project: ProjectGet


class RunnableGetDetailed(RunnableGet):
    expected_inputs: List[RunnableIOGet] = []
    expected_outputs: List[RunnableIOGet] = []
    code_excerpt: Optional[str] = None
    last_runs = []


# NOTE QUESTION: do we use these classes?
class FunctionGet(RunnableGet):
    type: RunnableType = Field(RunnableType.function, const=True)


class FunctionGetDetailed(FunctionGet, RunnableGetDetailed):
    pass


class PipelineGet(RunnableGet):
    type: RunnableType = Field(RunnableType.pipeline, const=True)


class PipelineGetDetailed(PipelineGet, RunnableGetDetailed):
    pass
