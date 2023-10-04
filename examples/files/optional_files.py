import cloudpickle as cp

from pipeline import File, Pipeline, Variable, pipe
from pipeline.cloud import environments, pipelines
from pipeline.configuration import current_configuration
from pipeline.objects.graph import InputField, InputSchema

current_configuration.set_debug_mode(True)


class MyInputSchema(InputSchema):
    in_1: File | None = InputField()


@pipe
def my_func(other_schema: MyInputSchema) -> str:
    if other_schema.in_1 is not None:
        # return other_schema.in_1.path.read_bytes()
        return cp.load(other_schema.in_1.path.open("rb"))

    return "No text found!"


with Pipeline() as builder:
    var_1 = Variable(MyInputSchema)

    output = my_func(var_1)

    builder.output(output)

my_pl = builder.get_pipeline()

# output = my_pl.run(MyInputSchema(in_1=File.from_object("Hello there")))
# print(output)

environments.create_environment(name="numpy", python_requirements=["numpy==1.25.0"])

remote_pipeline = pipelines.upload_pipeline(my_pl, "optional_file_test", "numpy")

output = pipelines.run_pipeline(
    remote_pipeline.id, MyInputSchema(in_1=File.from_object("H"))
)

print(output.result)
