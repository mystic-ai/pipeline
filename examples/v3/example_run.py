import datetime

from pipeline.v3 import run_pipeline

start_time = datetime.datetime.now()

result = run_pipeline("5", "my_data")

end_time = datetime.datetime.now()

total_time = (end_time - start_time).total_seconds()

print(f"Total time taken: {total_time}, result: '{result}")
