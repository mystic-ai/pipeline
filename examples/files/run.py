from pipeline.cloud.pipelines import run_pipeline

# Option 1
output = run_pipeline("file_test:v3", open("my_file.txt", "rb"))
print(output.result.result_array())

# Option 2
# run_pipeline(remote_pipeline.id, File("my_file.txt"))
# Option 3
# run_pipeline(remote_pipeline.id, FileURL("https://mystic.ai/storage/my_file.txt"))
