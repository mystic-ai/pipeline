import datetime

from pipeline.v3.pipelines import run_pipeline

start_time = datetime.datetime.now()

result = run_pipeline("1", 1)

end_time = datetime.datetime.now()

total_time = (end_time - start_time).total_seconds()

print(f"Total time taken: {total_time}, result: '{result}'")
