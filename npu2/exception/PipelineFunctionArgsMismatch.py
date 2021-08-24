class PipelineFunctionArgsMismatch(Exception):
    def __init__(self, func, next_func):
        self.func = func
        self.next_func = next_func
        output_args = None
        input_args = None

        if hasattr(func, "__npu_outputs__"):
            output_args = func.__npu__outputs__
        if hasattr(next_func, "__npu_inputs__"):
            input_args = next_func.__npu__inputs__

        super().__init__("First function: (name: '%s', output_args:%s) Second function: (name: '%s', input_args:%s)" % (func.__name__, output_args, next_func.__name__, input_args))