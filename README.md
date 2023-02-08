# [Pipeline](https://pipeline.ai) [![Version](https://img.shields.io/pypi/v/pipeline-ai)](https://pypi.org/project/pipeline-ai) ![Size](https://img.shields.io/github/repo-size/neuro-ai-dev/pipeline) ![Downloads](https://img.shields.io/pypi/dm/pipeline-ai) [![License](https://img.shields.io/crates/l/ap)](https://www.apache.org/licenses/LICENSE-2.0) [![Discord](https://img.shields.io/badge/discord-join-blue)](https://discord.gg/eJQRkBdEcs)

[_powered by mystic_](https://www.mystic.ai/)

# Table of Contents

- [About](#about)
- [Version roadmap](#version-roadmap)
  - [v0.4.0](#v040-jan-2023)
  - [v0.5.0](#v050-janfeb-2023)
  - [Beyond](#beyond)
- [Quickstart](#quickstart)
  - [Basic maths](#basic-maths)
  - [Transformers (GPT-Neo 125M)](#transformers-gpt-neo-125m)
- [Installation instructions](#installation-instructions)
  - [Linux, Mac (intel)](#linux--mac--intel-)
  - [Mac (arm/M1)](#mac--arm-m1-)
- [Development](#development)
- [License](#license)

# About

Pipeline is a python library that provides a simple way to construct computational graphs for AI/ML. The library is suitable for both development and production environments supporting inference and training/finetuning. This library is also a direct interface to [Pipeline.ai](https://pipeline.ai) which provides a compute engine to run pipelines at scale and on enterprise GPUs.

The syntax used for defining AI/ML pipelines shares some similarities in syntax to sessions in [Tensorflow v1](https://www.tensorflow.org/api_docs/python/tf/compat/v1/InteractiveSession), and Flows found in [Prefect](https://github.com/PrefectHQ/prefect). In future releases we will be moving away from this syntax to a C based graph compiler which interprets python directly (and other languages) allowing users of the API to compose graphs in a more native way to the chosen language.

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

# Quickstart

> :warning: **Uploading pipelines to Pipeline Cloud works best in Python 3.9.** We strongly recommend you use Python 3.9 when uploading pipelines because the `pipeline-ai` library is still in beta and is known to cause opaque errors when pipelines are serialised from a non-3.9 environment.

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

# Installation instructions

## Linux, Mac (intel)

```shell
pip install -U pipeline-ai
```

## Mac (arm/M1)

Due to the ARM architecture of the M1 core it is necessary to take additional steps to install Pipeline, mostly due to the transformers library. We recoomend running inside of a conda environment as shown below.

1. Make sure Rosetta2 is disabled.
2. From terminal run:

```
xcode-select --install
```

3. Install Miniforge, instructions here: [https://github.com/conda-forge/miniforge](https://github.com/conda-forge/miniforge) or follow the below:
   1. Download the Miniforge install script here: [https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh)
   2. Make the shell executable and run
   ```
   sudo chmod 775 Miniforge3-MacOSX-arm64.sh
   ./Miniforge3-MacOSX-arm64.sh
   ```
4. Create a conda based virtual env and activate:

```
conda create --name pipeline-env python=3.9
conda activate pipeline-env
```

5. Install tensorflow

```
conda install -c apple tensorflow-deps
python -m pip install -U pip
python -m pip install -U tensorflow-macos
python -m pip install -U tensorflow-metal
```

6. Install transformers

```
conda install -c huggingface transformers -y
```

7. Install pipeline

```
python -m pip install -U pipeline-ai
```

# Development

This project is made with poetry, [so firstly setup poetry on your machine](https://python-poetry.org/docs/#installation).

Once that is done, please run

    sh setup.sh

With this you should be good to go. This sets up dependencies, pre-commit hooks and
pre-push hooks.

You can manually run pre commit hooks with

    pre-commit run --all-files

To run tests manually please run

    pytest

# License

Pipeline is licensed under [Apache Software License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
