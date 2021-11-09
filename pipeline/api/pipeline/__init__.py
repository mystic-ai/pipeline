from pipeline.api.call import post

from pipeline.pipeline_schemas.pipeline import PipelineCreateSchema


def upload_pipeline(pipeline_create_schema: PipelineCreateSchema):
    request_result = post("/v2/pipelines", pipeline_create_schema.dict())

    return request_result
