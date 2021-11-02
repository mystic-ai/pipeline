import inspect

from npu2.object.Object import Object

from npu2.exception import InvalidObject
from npu2.function.save_function import save_function
from npu2.function.load_function import load_function

def __carry_npu_args__(func, execute_func):
    for arg in dir(func):
        if arg.startswith("__npu_"):
            setattr(execute_func, arg, getattr(func, arg))

def __validate_io_object__(object):
    if not inspect.isclass(object) or not issubclass(object, Object):
        raise InvalidObject(object)

def __validate_io_object_dict__(object_dict: dict):
    for io_object in object_dict.values():
        __validate_io_object__(io_object)

def __validate_function__(func):
    if hasattr(func, "__npu_inputs__"):
        __validate_io_object_dict__(func.__npu_inputs__)
    if hasattr(func, "__npu_outputs__"):
        __validate_io_object_dict__(func.__npu_outputs__)

def function(inputs={}, outputs={}):
    def inner_wrapper(func):
        def execute_func(*args, **kwargs):
            return func(*args, **kwargs)

        __carry_npu_args__(func, execute_func)
        
        execute_func.__npu_func__ = func

        # Handle decorator inputs
        __validate_io_object_dict__(inputs)
        execute_func.__npu_inputs__ = inputs
        
        # Handle decorator outputs
        __validate_io_object_dict__(outputs)
        execute_func.__npu_outputs__ = outputs
        
        return execute_func

    return inner_wrapper