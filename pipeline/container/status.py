from fastapi import APIRouter

router = APIRouter(prefix="/status")


@router.get(
    "",
    tags=["status"],
    status_code=200,
)
async def alive_check():
    return
