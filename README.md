# [Pipeline](https://pipeline.ai) [![Version](https://img.shields.io/pypi/v/pipeline-ai)](https://pypi.org/project/pipeline-ai) ![Size](https://img.shields.io/github/repo-size/neuro-ai-dev/pipeline) ![Downloads](https://img.shields.io/pypi/dm/pipeline-ai) [![License](https://img.shields.io/crates/l/ap)](https://www.apache.org/licenses/LICENSE-2.0) [![Discord](https://img.shields.io/badge/discord-join-blue)](https://discord.gg/eJQRkBdEcs)

[_powered by mystic_](https://www.mystic.ai/)

# Table of Contents

- [About](#about)
- [New to Pipeline?](#new-to-pipeline)
- [Quickstart](#quickstart)
  - [Hello pipeline](#hello-pipeline)
  - [Basic maths](#basic-maths)
  - [Transformers (GPT-Neo 125M)](#transformers-gpt-neo-125m)
- [Next steps?](#next-steps)
- [Installation instructions](#installation-instructions)
  - [Linux, Mac (intel)](#linux-mac-intel)
  - [Mac (arm/M1)](#mac-armm1)
  - [Development](#development)
- [Version roadmap](#version-roadmap)

  - [v0.4.0](#v040-jan-2023)
  - [v0.5.0](#v050-janfeb-2023)
  - [Beyond](#beyond)

- [License](#license)

# About

Pipeline is a python SDK that allows you to execute your local code anywhere.

It is primarily designed for ML engineers that want to run their code remotely without having to do any DevOps.
The library gives you the tools to create a compute graph that can be run at scale without needing to set up containers or kubernetes.
It is suitable for both development and production environments.
The Pipeline SDK interfaces directly with [Pipeline Cloud](https://pipeline.ai), which provides a compute engine to run pipelines at scale and on enterprise GPUs.

The syntax used for defining AI/ML pipelines shares some similarities with sessions in [Tensorflow v1](https://www.tensorflow.org/api_docs/python/tf/compat/v1/InteractiveSession), and Flows found in [Prefect](https://github.com/PrefectHQ/prefect).
In future releases we will be moving away from this syntax to a C based graph compiler which interprets python directly (and other languages) allowing users of the API to compose graphs in a more native way to the chosen language.

# New to Pipeline?

If you're ready to dive in and learn Pipeline, check out [the docs](https://docs.pipeline.ai) where you can try out the tutorials and create some fully-fledged Pipeline Cloud deployments, or read on for a few quick samples of Pipeline.

# Quickstart

Install `pipeline` with

```shell
pip install -U pipeline-ai
```

See the [installation guide](#installation-instructions) below for more detailed instructions.

## Hello pipeline

Import `Pipeline`, `pipeline_function` and decorate your Python function using the @pipeline_function decorator.

```python
from pipeline import Pipeline, pipeline_function

@pipeline_function
def hello_pipeline() -> str:
  	return "Hello Pipeline"
```

Configure your pipeline to use your Python functions through the `Pipeline` context manager:

```python
with Pipeline("hello-pipeline") as pipeline_builder:
  	greeting = hello_pipeline()
    pipeline_builder.output(greeting)
```

Then `get` the pipeline and run it:

```python
hello_pipeline = Pipeline.get_pipeline("hello-pipeline")
print(hello_pipeline.run())
```

And there you go, you're function is now being executed within a pipeline!

## Basic maths

```python
from pipeline import Pipeline, Variable, pipeline_function


@pipeline_function
def square(a: float) -> float:
    return a**2

@pipeline_function
def multiply(a: float, b: float) -> float:
    return a * b

with Pipeline("maths") as pipeline:
    flt_1 = Variable(type_class=float, is_input=True)
    flt_2 = Variable(type_class=float, is_input=True)
    pipeline.add_variables(flt_1, flt_2)

    sq_1 = square(flt_1)
    res_1 = multiply(flt_2, sq_1)
    pipeline.output(res_1)

output_pipeline = Pipeline.get_pipeline("maths")
print(output_pipeline.run(5.0, 6.0))

```

## Transformers (GPT-Neo 125M)

_Note: requires `torch` and `transformers` as dependencies._

```python
from pipeline import Pipeline, Variable
from pipeline.objects.huggingface.TransformersModelForCausalLM import (
    TransformersModelForCausalLM,
)

with Pipeline("hf-pipeline") as builder:
    input_str = Variable(str, is_input=True)
    model_kwargs = Variable(dict, is_input=True)

    builder.add_variables(input_str, model_kwargs)

    hf_model = TransformersModelForCausalLM(
        model_path="EleutherAI/gpt-neo-125M",
        tokenizer_path="EleutherAI/gpt-neo-125M",
    )
    hf_model.load()
    output_str = hf_model.predict(input_str, model_kwargs)

    builder.output(output_str)

output_pipeline = Pipeline.get_pipeline("hf-pipeline")

print(
    output_pipeline.run(
        "Hello my name is", {"min_length": 100, "max_length": 150, "temperature": 0.5}
    )
)
```

# Next Steps

Try our [cloud quickstart](https://docs.pipeline.ai/docs/cloud-quickstart) guide to learn about how to run pipelines in the cloud on enterprise GPUs. Then check out our tutorials for guides on how to deploy your own [custom pipelines](https://docs.pipeline.ai/docs/custom-models), [environments](https://docs.pipeline.ai/docs/custom-environments) and [integrate with third party services](https://docs.pipeline.ai/docs/integrations). Go deeper and learn about object concepts in the [Python API section](https://docs.pipeline.ai/docs/pipelinepipeline).

# Installation instructions

> :warning **Uploading pipelines to Pipeline Cloud works best in Python 3.9**.
> We strongly recommend you use Python 3.9 because the pipeline-ai library is still in beta and is known to cause opaque errors when pipelines are serialised from a non-3.9 environment.

We recommend installing Pipeline using a Python virtual environment manager such as `conda`, `poetry`, `pipenv` or `virtualenv/venv`.

## Linux, Mac (intel)

```shell
pip install -U pipeline-ai
```

## Mac (arm/M1)

Due to the ARM architecture of the M1 core it is necessary to take additional steps to install Pipeline, mostly due to the transformers library.
We recomend running inside of a `conda` environment as shown below.

1. Make sure `Rosetta2` is disabled.
2. From terminal run:

```shell
xcode-select --install
```

3. Install `Miniforge`, instructions here: <https://github.com/conda-forge/miniforge> or follow the below:
   1. Download the `Miniforge` install script here: <https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh>
   2. Make the shell executable and run
   ```shell
   sudo chmod 775 Miniforge3-MacOSX-arm64.sh
   ./Miniforge3-MacOSX-arm64.sh
   ```
4. Create a `conda` based virtual env and activate:

```shell
conda create --name pipeline-env python=3.9
conda activate pipeline-env
```

5. Install `tensorflow`

```shell
conda install -c apple tensorflow-deps
python -m pip install -U pip
python -m pip install -U tensorflow-macos
python -m pip install -U tensorflow-metal
```

6. Install `transformers`

```shell
conda install -c huggingface transformers -y
```

7. Install `pipeline`

```shell
python -m pip install -U pipeline-ai
```

## Versions

We strongly recommend installing the latest stable version of `pipeline`, but if you think you need to install a specific version, specify the version, such as:

```shell
pip install -U "pipeline-ai==0.4.6"
```

Find the available release versions in the [PyPI release history](https://pypi.org/project/pipeline-ai/#history).

## Development

This project is made with poetry, so to install Pipeline for development, firstly setup poetry on your machine.

Once that is done, run:

```shell
sh setup.sh
```

This sets up dependencies, pre-commit hooks and pre-push hooks. You should be good to go!

You can manually run pre commit hooks with:

```shell
pre-commit run --all-files
```

To run tests manually, run:

```shell
pytest
```

# Version roadmap

## v0.4.0 (Jan 2023)

- Custom environments on PipelineCloud (remote compute services)
- Kwarg inputs to runs
- Extended IO inputs to `pipeline_function` objects

## v0.5.0 (Jan/Feb 2023)

- Pipeline chaining
- `if` statements & `while/for` loops

## Beyond

- Run log streaming
- Run progress tracking
- Resource dedication
- Pipeline scecific remote load balancer (10% of traffic to one pipeline 80% to another)
- Usage capping
- Run result streaming
- Progromatic autoscaling
- Alerts
- Events
- Different python versions on remote compute services

# License

Pipeline is licensed under [Apache Software License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
