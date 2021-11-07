# Example: Basic maths

In this example we will implement successive mathematical functions into a Pipeline.
The source code for this example is found in `basic_maths.py` and at the end of this README.

## Imports

```
from pipeline import Pipeline, pipeline_function, Variable
```

`Pipeline` - The main class used to later define our compute pipeline.

`pipeline_function` - A decorator used to mark our pipeline functions. If this is not used on a function then the function will not be correctly processed with the Pipeline API.

`Variable` - Used to define input variables for our pipeline.

## Maths functions

The following functions are used later in our pipeline:

```
@pipeline_function
def square(a: float) -> float:
    return a ** 2


@pipeline_function
def minus(a: float, b: float) -> float:
    return a - b


@pipeline_function
def multiply(a: float, b: float) -> float:
    return a * b
```

As described in Imports, the `pipeline_decorator` ensures that the wrapped function correctly parses input/output data when defining our pipeline later. This decorator observes the _typing_ syntax used for each individual variable in the function, including the output.

## Pipeline definition

## Source

```
from pipeline import Pipeline, pipeline_function, Variable


@pipeline_function
def square(a: float) -> float:
    return a ** 2


@pipeline_function
def minus(a: float, b: float) -> float:
    return a - b


@pipeline_function
def multiply(a: float, b: float) -> float:
    return a * b


with Pipeline("MathsIsFun") as pipeline:
    flt_1 = Variable(variable_type=float, is_input=True)
    flt_2 = Variable(variable_type=float, is_input=True)

    sq_1 = square(flt_1)
    res_1 = multiply(flt_2, sq_1)
    res_2 = minus(res_1, sq_1)
    sq_2 = square(res_2)
    res_3 = multiply(flt_2, sq_2)
    res_4 = minus(res_3, sq_1)
    pipeline.output(res_2, res_4)

output_pipeline = Pipeline.get_pipeline("MathsIsFun")
print(output_pipeline.run(5.0, 6.0))

```
