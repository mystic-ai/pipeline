import logging
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, UploadFile, status

logger = logging.getLogger("uvicorn")
router = APIRouter(prefix="/files", tags=["Files"])


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
)
async def file_upload(
    pfile: UploadFile,
):
    """Upload a file to the container. Returns the path of the file on the container."""
    data = await pfile.read()
    pfile_uid = str(uuid4())
    name = getattr(pfile, "filename")
    name = Path(name).name

    logger.info(f"Received file: {name}")
    pfile_path = Path(f"/tmp/{pfile_uid[:2]}/{pfile_uid[2:4]}/{name}")
    pfile_path.parent.mkdir(parents=True, exist_ok=True)

    with open(pfile_path, "wb") as f:
        f.write(data)

    return dict(path=str(pfile_path))
