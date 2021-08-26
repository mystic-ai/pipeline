from npu2 import pipeline
from npu2.exception import PipelineFunctionOutputFormatError

class Pipeline:
    name: str
    pipeline_array: list

    def __init__(self, name="", pipeline_array=[]):
        self.pipeline_array = pipeline_array
        self.name = name


    def __handle_function_output__(self, func, function_output):
        if isinstance(function_output,  tuple):
            if len(function_output) == 2:
                if isinstance(function_output[1], dict):
                    # Assume this is formatted correctly and the user isn't being silly. 
                    # TODO: Make this better
                    return function_output[0], function_output[1]
            
            raise PipelineFunctionOutputFormatError(func, function_output)

    def run(self, *args, **kwargs):
        output_args = None
        outputs = self.pipeline_array[0](*args, **kwargs)
        output_args, output_kwargs = self.__handle_function_output__(self.pipeline_array[0], outputs)

        for func in self.pipeline_array[1:]:
            outputs = func(*output_args, **output_kwargs)
            output_args, output_kwargs = self.__handle_function_output__(func, outputs)
            
        return output_args, output_kwargs