import logging

from fastapi import APIRouter

from pipeline.cloud.schemas import runs as run_schemas

logger = logging.getLogger("uvicorn")
router = APIRouter(prefix="/pipeline")


# @router.get("/cache", tags=["pipeline"], status_code=200, response_model=run_schemas.Run)
# async def run(run_create: run_schemas.RunCreate):
#     return
