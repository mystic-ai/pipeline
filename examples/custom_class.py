from pipeline import Paiplain
from pipeline.objects.function import pipeline_function
from pipeline.objects.pipeline import Pipeline
from pipeline.objects.variable import Variable


class MyClass(object):
    def __init__(self, var):
        self.var = var


@pipeline_function
def add_lol(a: str) -> MyClass:
    return MyClass(a + " lol")


with Pipeline() as pipeline:
    my_class_var = Variable(type_class=str, is_input=True)
    pipeline.add_variable(my_class_var)

    output_class = add_lol(my_class_var)
    print(output_class)

    pipeline.output(output_class)


print(pipeline.run("Hey")[0].var)

########################################################################
#                 example of proposal for comparisson                  #
########################################################################


class YourClass(object):
    def __init__(self, var):
        self.var = var


pipeline = Paiplain("custom_class")


@pipeline.stage
def add_lol(a: str) -> YourClass:
    return YourClass(a + " lol")


print(pipeline.run("Hey")[0].var)
