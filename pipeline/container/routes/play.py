import pkg_resources
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(prefix="/play", tags=["play"])


@router.get("", response_class=HTMLResponse)
async def render_pipeline_play():
    ts_code = pkg_resources.resource_string(
        "pipeline", "container/frontend/index.html"
    ).decode("utf-8")

    return f"""{ts_code}"""
