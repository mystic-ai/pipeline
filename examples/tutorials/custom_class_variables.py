from pipeline import Pipeline, Variable, pipeline_function


class MyClass:
    def __init__(self, var):
        self.var = var


@pipeline_function
def add_lol(a: str) -> MyClass:
    return MyClass(a + " lol")


with Pipeline("custom_class") as pipeline:
    my_class_var = Variable(type_class=str, is_input=True)
    pipeline.add_variable(my_class_var)

    output_class = add_lol(my_class_var)

    pipeline.output(output_class)

output_pipeline = Pipeline.get_pipeline("custom_class")
print(output_pipeline.run("Hey")[0].var)
