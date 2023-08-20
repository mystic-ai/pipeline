from pipeline import File, Pipeline, Variable, entity, pipe
from pipeline.cloud.pipelines import upload_pipeline


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

    my_file = File.from_remote(id="file_e5a7000de4564edb9678904f60e75c02")
    en = MainClass()

    en.my_func(my_file)
    res = en.get_text(rand)
    builder.output(res)

my_pl = builder.get_pipeline()


remote_pipeline = upload_pipeline(my_pl, "remote-file-test", "numpy")
