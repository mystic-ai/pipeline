import datetime

from pipeline.cloud.pipelines import poll_async_run, run_pipeline

pointer = "ph/test:test"
print(f"Running test pipeline async: {pointer}")

start_time = datetime.datetime.now()

result = run_pipeline(
    pointer,
    5,
    async_run=True,
)

result = poll_async_run(
    result.id,
    timeout=None,
    interval=0.5,
)

end_time = datetime.datetime.now()

total_time = (end_time - start_time).total_seconds() * 1e3

print(
    "Total time taken: %.3f ms, result: '%s'"
    % (
        total_time,
        result.result.result_array(),
    )
)
