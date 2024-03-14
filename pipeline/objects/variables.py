from typing import Generic, Iterable, TypeVar

ST = TypeVar("ST")


class Stream(Generic[ST], Iterable):
    def __init__(self, iterable: Iterable[ST]):
        self.iterable = iterable

    def __iter__(self):
        return self.iterable.__iter__()

    def __next__(self):
        return self.iterable.__next__()
