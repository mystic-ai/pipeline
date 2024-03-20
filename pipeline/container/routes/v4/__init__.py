from fastapi import APIRouter

from pipeline.container.routes.v4.container import router as v4_container_router
from pipeline.container.routes.v4.files import router as v4_file_router
from pipeline.container.routes.v4.runs import router as v4_runs_router

router = APIRouter(prefix="/v4")

router.include_router(v4_runs_router)
router.include_router(v4_file_router)
router.include_router(v4_container_router)
