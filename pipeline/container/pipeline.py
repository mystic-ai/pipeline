import logging

from fastapi import APIRouter

logger = logging.getLogger("uvicorn")
router = APIRouter(prefix="/pipeline")
