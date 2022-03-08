from pipeline import Pipeline, Variable, pipeline_function


@pipeline_function
def multiply(a: float, b: float) -> float:
    return a * b


with Pipeline("MathsIsFun") as builder:
    flt_1 = Variable(float, is_input=True)
    flt_2 = Variable(float, is_input=True)
    builder.add_variables(flt_1, flt_2)
    multiply_result = multiply(flt_1, flt_2)

    builder.output(multiply_result)

output_pipeline = Pipeline.get_pipeline("MathsIsFun")
print(output_pipeline.run(5.0, 6.0))
