from pipeline import Pipeline, Variable, entity, pipe
from pipeline.cloud.pipelines import upload_pipeline
from pipeline.objects import Directory


@entity
class MainClass:
    @pipe(run_once=True, on_startup=True)
    def my_func(self, my_file: Directory) -> str:
        self.path = my_file.path

    @pipe
    def get_text(self, rand: str) -> str:
        child_files = self.path.glob("*")
        return str(list(child_files))


with Pipeline() as builder:
    rand = Variable(str)

    my_file = Directory.from_remote(id="file_e2e16eb2af124fc1a369eaef832a8792")
    en = MainClass()

    en.my_func(my_file)
    res = en.get_text(rand)
    builder.output(res)

my_pl = builder.get_pipeline()


remote_pipeline = upload_pipeline(my_pl, "remote-directory-test", "numpy")
