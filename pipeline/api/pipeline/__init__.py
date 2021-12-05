from pipeline.api.call import post

from pipeline.schemas.pipeline import PipelineCreate


def upload_pipeline(pipeline_create_schema: PipelineCreate):
    request_result = post("/v2/pipelines", pipeline_create_schema.dict())

    return request_result
