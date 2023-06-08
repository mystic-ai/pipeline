import datetime

from pipeline.v3.pipelines import run_pipeline

start_time = datetime.datetime.now()

result = run_pipeline("pipeline_0edac51784f74150ab921cca6e34ccdf", 1)

end_time = datetime.datetime.now()

total_time = (end_time - start_time).total_seconds()

print(f"Total time taken: {total_time}, result: '{result}'")
