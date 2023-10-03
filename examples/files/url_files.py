from pipeline import File, Pipeline, Variable, pipe
from pipeline.cloud import environments, pipelines
from pipeline.configuration import current_configuration

current_configuration.set_debug_mode(True)


@pipe
def my_func(file: File) -> File:
    return file


with Pipeline() as builder:
    var_1 = Variable(File)

    output = my_func(var_1)

    builder.output(output)

my_pl = builder.get_pipeline()

# output = my_pl.run(MyInputSchema(in_1=File.from_object("Hello there")))
# print(output)

environments.create_environment(name="numpy", python_requirements=["numpy==1.25.0"])

remote_pipeline = pipelines.upload_pipeline(my_pl, "file_url_test", "numpy")

url = "https://storage.mystic.ai/run_files/85/cd/85cdb894-df0b-4a06-8200-ca0cda6bd163/image-0.jpg"  # noqa

output = pipelines.run_pipeline(
    remote_pipeline.id,
    File(
        url=url,
    ),
)

output.outputs()[0].save("./test.jpg")

...
# output.result.outputs[0].save("./test.jpg")
