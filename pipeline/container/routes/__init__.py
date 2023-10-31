from fastapi import APIRouter

from pipeline.container.routes.v4 import router as v4_router

router = APIRouter()
router.include_router(v4_router)
