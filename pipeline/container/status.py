import logging

from fastapi import APIRouter

logger = logging.getLogger("uvicorn")
router = APIRouter(prefix="/status")


@router.get(
    "",
    tags=["status"],
    status_code=200,
)
async def alive_check():
    return
