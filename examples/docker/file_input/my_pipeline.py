import os

from pipeline import Pipeline, Variable, pipe
from pipeline.objects.graph import File, InputField, InputSchema


class FileSchema(InputSchema):
    file_1: File | None = InputField()


@pipe
def get_file_size(my_file: File, file_schema: FileSchema) -> int:
    file_size = os.path.getsize(my_file.path)
    if file_schema.file_1 is not None:
        return file_size + os.path.getsize(file_schema.file_1.path)


with Pipeline() as builder:
    file = Variable(
        type_class=File,
        title="input-file",
        description="The input file whose size will be computed.",
    )
    params = Variable(
        FileSchema,
        title="params",
        description="Additional parameters to the model",
    )

    output = get_file_size(file, params)

    builder.output(output)

pipeline_graph = builder.get_pipeline()
