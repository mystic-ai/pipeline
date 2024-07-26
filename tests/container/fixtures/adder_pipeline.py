import logging

from pipeline import Pipeline, Variable, pipe

logger = logging.getLogger(__name__)


@pipe
def add(first: int, second: int) -> int:
    return first + second


with Pipeline() as builder:
    first = Variable(int)
    second = Variable(int)
    result = add(first, second)
    builder.output(result)

my_pipeline = builder.get_pipeline()
