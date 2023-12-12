import typing as t

from fastapi import APIRouter, Request, Response

from pipeline.cloud.schemas import pipelines as pipeline_schemas
from pipeline.container.manager import Manager
from pipeline.objects import Graph

router = APIRouter(prefix="/container")


@router.get(
    "/state",
    tags=["container"],
    status_code=200,
    response_model=pipeline_schemas.PipelineContainerState,
    responses={
        503: {
            "description": "Pipeline is loading",
            "model": pipeline_schemas.PipelineContainerState,
        },
        500: {
            "description": "Pipeline failed to load",
            "model": pipeline_schemas.PipelineContainerState,
        },
    },
)
async def is_ready(request: Request, response: Response):
    run_manager = request.app.state.manager
    if run_manager.pipeline_state in [
        pipeline_schemas.PipelineState.loading,
        pipeline_schemas.PipelineState.not_loaded,
    ]:
        response.status_code = 503
    elif run_manager.pipeline_state == pipeline_schemas.PipelineState.failed:
        response.status_code = 500

    return pipeline_schemas.PipelineContainerState(
        state=run_manager.pipeline_state,
        message=run_manager.pipeline_state_message,
    )


@router.get(
    "/pipeline",
    tags=["container"],
    response_model=pipeline_schemas.Pipeline,
)
async def get_pipeline(request: Request):
    run_manager: Manager = request.app.state.manager
    input_variables: t.List[pipeline_schemas.IOVariable] = []
    output_variables: t.List[pipeline_schemas.IOVariable] = []
    graph: Graph = run_manager.pipeline
    for variable in graph.variables:
        if variable.is_input:
            input_variables.append(variable.to_io_schema())

        if variable.is_output:
            output_variables.append(variable.to_io_schema())

    input_variables = input_variables
    output_variables = output_variables
    return pipeline_schemas.Pipeline(
        name=run_manager.pipeline_name,
        image=run_manager.pipeline_image,
        input_variables=input_variables,
        output_variables=output_variables,
    )
