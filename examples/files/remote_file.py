from pipeline import File, Pipeline, Variable, entity, pipe
from pipeline.cloud.pipelines import run_pipeline, upload_pipeline
from pipeline.configuration import current_configuration

current_configuration.set_debug_mode(True)


@entity
class MainClass:
    @pipe(run_once=True, on_startup=True)
    def my_func(self, my_file: File) -> str:
        self.text = my_file.path.read_text()

    @pipe
    def get_text(self, rand: str) -> str:
        return self.text


with Pipeline() as builder:
    rand = Variable(str)

    my_file = File(remote_id="file_541488b627e04ab198ecb79b2e81b53a")
    en = MainClass()

    en.my_func(my_file)
    res = en.get_text(rand)
    builder.output(res)

my_pl = builder.get_pipeline()


remote_pipeline = upload_pipeline(my_pl, "remote-file-test", "numpy")

output = run_pipeline(
    remote_pipeline.id,
    "lol",
)

print(output.outputs())
