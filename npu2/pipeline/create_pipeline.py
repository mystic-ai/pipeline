import inspect

from npu2.exception import PipelineFunctionArgsMismatch
from npu2.function import __validate_function__

def create_pipeline(pipeline_array: list):
    for func in pipeline_array:
        __validate_function__(func)
    
    for index, func in enumerate(pipeline_array):
        if index < len(pipeline_array) - 1:
            # Check check if the next stage get the necessary things from this stage, the outputs must match inputs, unless kwarg
            next_func = pipeline_array[index + 1]

            # TODO: This feels like trash, find a better way to do kwargs?
            next_func_argspec = inspect.getargspec(next_func)
            next_func_default_kwargs_inputs = next_func_argspec.defaults
            next_func_kwargs = next_func_argspec.args[(len(next_func_argspec.args) - len(next_func_argspec.defaults)):]
            next_func_args = next_func_argspec.args[:(len(next_func_argspec.args) - len(next_func_argspec.defaults))]

            if ((not hasattr(func, "__npu_outputs__")) and hasattr(next_func, "__npu_inputs__")) \
                or (hasattr(func, "__npu_outputs__") and (not hasattr(next_func, "__npu_inputs__"))):
                raise PipelineFunctionArgsMismatch(func, next_func)
            elif not hasattr(func, "__npu_outputs__") and not hasattr(next_func, "__npu_inputs__"):    
                continue
            
            # TODO: I'm ashamed of the following lines, find a better way to do it

            func_outputs = func.__npu_outputs__
            next_func_inputs = next_func.__npu_inputs__

            if len(func_outputs) > len(next_func_inputs):
                raise PipelineFunctionArgsMismatch(func, next_func)

            for output in func_outputs.keys():
                if func_outputs[output] == next_func_inputs[output]:
                    func_outputs.pop(output, None)
                    next_func_inputs.pop(output,None)
                else:
                    raise PipelineFunctionArgsMismatch(func, next_func)
            
            if len(func_outputs) > 0:
                raise PipelineFunctionArgsMismatch(func, next_func)

            for input in next_func_inputs.keys():
                if input in next_func_kwargs:
                    next_func_inputs.pop(input,None)

            if len(next_func_inputs) > 0:
                raise PipelineFunctionArgsMismatch(func, next_func)