import io

from typing import Union, Any

from pipeline.api.call import post
from pipeline.api.file import upload_file

from pipeline.util import python_object_to_hex

from pipeline.schemas.pipeline import PipelineGet
from pipeline.schemas.run import RunCreate


def run_pipeline(
    pipeline_id_or_schema: Union[str, PipelineGet], data_or_file_id: Union[Any, str]
):
    # TODO: Add support for generic object inference. Only strs at the moment.
    file_id = None
    if isinstance(data_or_file_id, str):
        file_id = data_or_file_id
    else:
        temp_file = io.BytesIO(python_object_to_hex(data_or_file_id).encode())
        uploaded_data = upload_file(temp_file, "/")
        file_id = uploaded_data.id

    pipeline_id = None
    if isinstance(pipeline_id_or_schema, str):
        pipeline_id = pipeline_id_or_schema
    elif isinstance(pipeline_id_or_schema, PipelineGet):
        pipeline_id = pipeline_id_or_schema.id
    else:
        raise Exception(
            "Must either pass a pipeline id, or a pipeline get schema. Not object of type %s in arg 1."
            % str(pipeline_id_or_schema)
        )

    # TODO: swap "data=data_or_file_id" for "file_id=file_id" later when the generic object inference is added back.
    run_create_schema = RunCreate(pipeline_id=pipeline_id, data=data_or_file_id)

    return post("/v2/runs", run_create_schema.dict())
