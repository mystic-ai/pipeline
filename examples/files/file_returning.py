from pathlib import Path
from typing import List

from pipeline import Pipeline, Variable, entity, pipe
from pipeline.cloud import compute_requirements, environments, pipelines
from pipeline.objects import File
from pipeline.objects.graph import InputField, InputSchema


@entity
class ReturnAFile:
    @pipe
    def get_file(self, random_string: str) -> File:
        file_path = "/tmp/random.txt"
        with open("/tmp/random.txt", "w") as f:
            f.write(random_string)
        output_image = File(path=file_path, allow_out_of_context_creation=True)

        return output_image


with Pipeline() as builder:
    prompt = Variable(str)

    model = ReturnAFile()

    output = model.predict(prompt)

    builder.output(output)

my_pl = builder.get_pipeline()
