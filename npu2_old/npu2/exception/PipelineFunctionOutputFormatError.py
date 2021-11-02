class PipelineFunctionOutputFormatError(Exception):
    def __init__(self, func, outputs):
        self.func = func
        self.next_func = outputs

        super().__init__("Output of an npu2 function must be in the following format tuple dict format (args, kwargs) to pass on to another function: (), {}, function name: %s, got outputs: %s" % (func.__name__, outputs))