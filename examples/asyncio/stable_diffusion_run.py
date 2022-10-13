import asyncio

from pipeline.api.asyncio import PipelineCloud

api = PipelineCloud()


async def run():
    res = await api.run_pipeline(
        "pipeline_67d9d8ec36d54c148c70df1f404b0369", [["a dog"], {}]
    )
    print(res, flush=True)


loop = asyncio.get_event_loop()
loop.run_until_complete(run())
