from __future__ import annotations

from typing import Any, Generic, List, TypeVar

from .base import BaseModel, GenericModel

DataType = TypeVar("DataType")


class PaginationDetails(BaseModel):
    skip: int
    limit: int


class Paginated(GenericModel, Generic[DataType]):

    skip: int
    limit: int
    total: int
    data: List[DataType]

    @classmethod
    def of(
        cls, item_list: List[DataType], details: PaginationDetails
    ) -> Paginated[Any]:

        return Paginated(
            skip=details.skip,
            limit=details.limit,
            total=len(item_list),
            data=item_list,
        )
