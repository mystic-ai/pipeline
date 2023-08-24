import typing as t

from pydantic import conint

from . import BaseModel, GenericModel

DataType = t.TypeVar("DataType")

PAGINATION_LIMIT = 1000


class Pagination(BaseModel):
    """Query parameters for requesting a set of items of a given resource,
    in paginated form."""

    #: Index of the first item to return within the total queryset
    skip: conint(ge=0)  # type: ignore

    #: Maximum number of items to return
    limit: conint(ge=1, le=PAGINATION_LIMIT)  # type: ignore


class Paginated(GenericModel, t.Generic[DataType]):
    """Response model for a paginated set of a given resource."""

    #: Index of the first item to return within the total queryset
    skip: conint(ge=0)  # type: ignore

    #: Maximum number of items the `data` field will contain
    limit: conint(ge=1, le=PAGINATION_LIMIT)  # type: ignore

    #: Number of items within the total queryset
    total: conint(ge=0)  # type: ignore

    #: Resource data, which should be immutable
    data: t.Sequence[DataType]

    @classmethod
    def of(cls, data: t.Sequence[DataType], pagination: Pagination, total: int):
        return Paginated(**pagination.dict(), total=total, data=data)
