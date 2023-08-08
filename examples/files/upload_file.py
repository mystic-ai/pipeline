from pipeline import File, Pipeline, Variable, pipe
from pipeline.cloud.pipelines import run_pipeline, upload_pipeline


@pipe
def my_func(my_file: File) -> str:
    return my_file.path.read_text()


with Pipeline() as builder:
    var_1 = Variable(File)

    output = my_func(var_1)

    builder.output(output)

my_pl = builder.get_pipeline()


remote_pipeline = upload_pipeline(my_pl, "file_test", "numpy")

# Option 1
output = run_pipeline(remote_pipeline.id, open("my_file.txt", "rb"))

print(output.result.result_array())
# Option 2
# run_pipeline(remote_pipeline.id, File("my_file.txt"))
# Option 3
# run_pipeline(remote_pipeline.id, FileURL("https://mystic.ai/storage/my_file.txt"))
