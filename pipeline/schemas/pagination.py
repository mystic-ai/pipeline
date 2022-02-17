from typing import Generic, List, TypeVar

from .base import BaseModel, GenericModel

DataType = TypeVar("DataType")


class PaginationDetails(BaseModel):
    """Query parameters for requesting paginated resource lists."""

    #: Index of resource items to begin paginating from
    skip: int
    #: Maximum number of resources to return
    limit: int


class Paginated(GenericModel, Generic[DataType]):
    """Response for paginated resource lists."""

    #: Index of resource items to begin paginating from
    skip: int
    #: Maximum number of resources the `data` field will contain
    limit: int
    #: Total number of resources available for pagination
    total: int
    #: Resource data
    data: List[DataType]

    @classmethod
    def of(cls, item_list: List[DataType], details: PaginationDetails, total: int):

        return Paginated(
            skip=details.skip, limit=details.limit, total=total, data=item_list
        )
