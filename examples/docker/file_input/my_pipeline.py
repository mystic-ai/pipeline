import os

from pipeline import Pipeline, Variable, pipe
from pipeline.objects.graph import File


@pipe
def get_file_size(my_file: File) -> int:
    return os.path.getsize(my_file.path)


with Pipeline() as builder:
    file = Variable(
        type_class=File,
        title="input-file",
        description="The input file whose size will be computed.",
    )

    output = get_file_size(file)

    builder.output(output)

pipeline_graph = builder.get_pipeline()
