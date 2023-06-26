from httpx import HTTPStatusError

from pipeline.api import PipelineCloud
from pipeline.v3.schemas import pointers as pointer_schemas


def create_pointer(
    pointer: str,
    target_pipeline_id_or_pointer: str,
    overwrite: bool = False,
    locked: bool = False,
) -> None:
    cluster_api = PipelineCloud(verbose=False)

    try:
        cluster_api._post(
            "/v3/pointers",
            json_data=pointer_schemas.PointerCreate(
                pointer=pointer,
                pointer_or_pipeline_id=target_pipeline_id_or_pointer,
                locked=locked,
            ).dict(),
        )
    except HTTPStatusError as e:
        if e.response.status_code == 409 and overwrite:
            cluster_api._patch(
                f"/v3/pointers/{pointer}",
                json_data=pointer_schemas.PointerPatch(
                    pointer_or_pipeline_id=target_pipeline_id_or_pointer,
                ).dict(),
            )
        elif e.response.status_code == 404:
            print(e.response.text)
            print(f"Pipeline {target_pipeline_id_or_pointer} not found")
        else:
            raise
