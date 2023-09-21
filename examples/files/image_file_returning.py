import urllib.request

from pipeline import Pipeline, Variable, entity, pipe
from pipeline.objects import File


@entity
class ReturnAFile:
    @pipe
    def get_file(self, random_string: str) -> File:
        file_path = "/tmp/image.jpg"
        urllib.request.urlretrieve(
            "https://storage.mystic.ai/run_files/31/7e/317e304d-e816-4036-86b2-7ad82b208b70/image-0.jpg",  # noqa
            file_path,
        )

        output_image = File(path=file_path)

        return output_image


with Pipeline() as builder:
    prompt = Variable(str)

    model = ReturnAFile()

    output = model.predict(prompt)

    builder.output(output)

my_pl = builder.get_pipeline()
