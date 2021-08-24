import inspect
import copy

from npu2.exception import PipelineFunctionArgsMismatch
from npu2.function import __validate_function__

from npu2.pipeline import Pipeline


def create_pipeline(pipeline_array: list, pipeline_name=""):
    for func in pipeline_array:
        __validate_function__(func)

    for index, func in enumerate(pipeline_array):
        if index < len(pipeline_array) - 1:
            # Check check if the next stage get the necessary things from this stage, the outputs must match inputs, unless kwarg
            next_func = pipeline_array[index + 1]

            # TODO: This feels like trash, find a better way to do kwargs?
            next_func_argspec = inspect.getargspec(next_func)
            next_func_default_kwargs_inputs = next_func_argspec.defaults

            next_func_kwargs = next_func_argspec.args[
                (len(next_func_argspec.args) - len(next_func_argspec.defaults)) :
            ] if next_func_default_kwargs_inputs != None else None

            next_func_args = next_func_argspec.args[
                : (len(next_func_argspec.args) - len(next_func_argspec.defaults))
            ] if next_func_default_kwargs_inputs != None else None

            if (
                (not hasattr(func, "__npu_outputs__"))
                and hasattr(next_func, "__npu_inputs__")
            ) or (
                hasattr(func, "__npu_outputs__")
                and (not hasattr(next_func, "__npu_inputs__"))
            ):
                raise PipelineFunctionArgsMismatch(func, next_func)
            elif not hasattr(func, "__npu_outputs__") and not hasattr(
                next_func, "__npu_inputs__"
            ):
                continue

            # TODO: I'm ashamed of the following lines, find a better way to do it

            func_outputs = func.__npu_outputs__
            next_func_inputs = next_func.__npu_inputs__

            func_outputs_copy = copy.deepcopy(func_outputs)
            next_func_inputs_copy = copy.deepcopy(next_func_inputs)

            if len(func_outputs) > len(next_func_inputs):
                raise PipelineFunctionArgsMismatch(func, next_func)

            for output in func_outputs.keys():
                if output in next_func_inputs and func_outputs[output] == next_func_inputs[output]:
                    func_outputs_copy.pop(output, None)
                    next_func_inputs_copy.pop(output, None)
                else:
                    raise PipelineFunctionArgsMismatch(func, next_func)

            func_outputs = func_outputs_copy
            next_func_inputs = next_func_inputs_copy

            if len(func_outputs) > 0:
                raise PipelineFunctionArgsMismatch(func, next_func)

            for input in next_func_inputs.keys():
                if input in next_func_kwargs:
                    next_func_inputs_copy.pop(input, None)
            
            next_func_inputs = copy.deepcopy(next_func_inputs_copy)

            if len(next_func_inputs) > 0:
                raise PipelineFunctionArgsMismatch(func, next_func)

    new_pipeline = Pipeline(pipeline_array=pipeline_array, name=pipeline_name)
    return new_pipeline