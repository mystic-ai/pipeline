---
title: Getting started
excerpt: A quick setup tutorial for the pipeline python library.
category: 61e6de6d35c07f00106bc18d
---
# Contents

- [What is a compute pipeline](#what--is--a--compute--pipeline)
  * [Basic Pipeline](#basic--pipeline)
    *  [Imports](#imports)
    *  [Function definition](#function--definition)
    *  [Context manager](#context--manager)


# What is a compute pipeline
In the Pipeline Library a compute pipeline describes computations that are applied to a series of inputs to produce a series of outputs. These computations are defined in either functions or other pipelines.

The Pipeline Library includes several features that are directed towards ML applications and productionaisation applications, but first we will look at a basic example.

## Basic pipeline
When defining Pipelines we do not want to run them, but save them for running later. The Pipeline Library uses a series of decorators to change the default behaviour of functions when used inside of a Context Manager (the `with ...` statement used below). When the context manager is active, all functions that are wrapped with the `pipeline_function` decorator do not actually execute but rather return a reference to a variable that is stored in the pipeline. Functions wrapped in the `pipeline_function` decorator will execute normally when used outside of the Pipeline context manager.

Below we have a simple example of multiplying two numbers together and returning the result:
```
from pipeline import Pipeline, Variable, pipeline_function


@pipeline_function
def multiply(a: float, b: float) -> float:
    return a * b


with Pipeline("MathsIsFun") as builder:
    # Define the inputs used to feed data into the pipeline
    flt_1 = Variable(float, is_input=True)
    flt_2 = Variable(float, is_input=True)
    # Add the variables to the pipeline
    builder.add_variables(flt_1, flt_2)

    # Perform a computation on the inputs
    multiply_result = multiply(flt_1, flt_2)

    # Use the computation output as the output for the whole pipeline
    builder.output(multiply_result)

output_pipeline = Pipeline.get_pipeline("MathsIsFun")
print(output_pipeline.run(5.0, 6.0))
# The output of this pipeline is 30.0
```



### Imports
The imports in this example will be consistent accross most uses of the library:
```
from pipeline import Pipeline, Variable, pipeline_function
```

Here the `Pipeline` is the main class used when we will interact with the pipeline builder later, with `Variable` and `pipeline_function` allowing us to define objects that will be used inside of the pipeline.

The `Variable` class is the same used internally in the pipeline for referencing the inputs/outputs of functions.

### Function definition

A function definition requires 2 things:
- `pipeline_function` decorator
- typing descriptors on inputs and outputs (`a: float, b: float` and the return type `-> float`)
```
@pipeline_function
def multiply(a: float, b: float) -> float:
    return a * b
```

As seen in this example the `multiply` function takes in two floats and returns a float. By using these typing descriptors the `pipeline_function` decorator to know what your function is expecting in and out; then in-turn ensure that it does. You can use any generic python class as the input and output type.

### Context manager
Now that we have defined what we need to use in our simple pipeline, we can define it:

```
with Pipeline("MathsIsFun") as builder:
    flt_1 = Variable(float, is_input=True)
    flt_2 = Variable(float, is_input=True)
    builder.add_variables(flt_1, flt_2)
    multiply_result = multiply(flt_1, flt_2)
    builder.output(multiply_result)
```

With the pipeline being defined this way are variables are correctly passed from input to output correclty. Below is an example of how not to do the computation:

```
# This will break!
with Pipeline("MathsIsFun") as builder:
    flt_1 = Variable(float, is_input=True)
    flt_2 = Variable(float, is_input=True)
    builder.add_variables(flt_1, flt_2)
    multiply_result = flt_1 * flt_2 # Here we directly do the computation in the context manager
    builder.output(multiply_result)
```

In the above example we attempt to perform the multiplication directly in the context manager. This will not work. Although the Variables `flt_1 flt_2` are defined as floats, they are not actually float objects. Here, python will try to multiply the two Variables together, but they are actually Variable objects (references).

