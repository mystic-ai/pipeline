import pkg_resources
from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(prefix="/play", tags=["play"])



@router.get("", response_class=FileResponse)
async def serve_root():
    file_path = pkg_resources.resource_filename("pipeline", "container/frontend/dist/index.html")
    return FileResponse(file_path)