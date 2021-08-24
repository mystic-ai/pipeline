from npu2 import pipeline


class Pipeline:
    name: str
    pipeline_array: list

    def __init__(self, name="", pipeline_array=[]):
        self.pipeline_array = pipeline_array
        self.name = name

    def run(self, *args, **kwargs):
        output_args = None

        output_args = self.pipeline_array[0](*args, **kwargs)

        for func in self.pipeline_array[1:]:
            output_args = func(**output_args)
            
